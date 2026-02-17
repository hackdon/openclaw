import type { IncomingMessage, ServerResponse } from "node:http";
import { completeSimple, type TextContent } from "@mariozechner/pi-ai";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import { resolveModel } from "../../src/agents/pi-embedded-runner/model.js";
import { getApiKeyForModel, requireApiKey } from "../../src/agents/model-auth.js";
import { resolveConfiguredModelRef } from "../../src/agents/model-selection.js";
import { resolveAgentDir } from "../../src/agents/agent-scope.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../../src/agents/defaults.js";
import { resolveGatewayAuth } from "../../src/gateway/auth.js";

const MAX_BODY_BYTES = 1024 * 1024;

// --- HTTP helpers (inline to avoid depending on internal gateway modules) ---

function getBearerToken(req: IncomingMessage): string | undefined {
  const raw = (req.headers.authorization ?? "").trim();
  if (!raw.toLowerCase().startsWith("bearer ")) return undefined;
  const token = raw.slice(7).trim();
  return token || undefined;
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}

function readBody(req: IncomingMessage, maxBytes: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > maxBytes) {
        req.destroy();
        reject(new Error("payload too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

// --- Request parsing ---

type ChatMessage = { role?: string; content?: unknown };

function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part?.type === "text" && typeof part.text === "string") return part.text;
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function parseModelRef(raw: string | undefined, defaultProvider: string, defaultModel: string) {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return { provider: defaultProvider, model: defaultModel };
  if (trimmed.includes("/")) {
    const [provider, ...rest] = trimmed.split("/");
    return { provider: provider!, model: rest.join("/") };
  }
  return { provider: defaultProvider, model: trimmed };
}

function isTextBlock(block: { type: string }): block is TextContent {
  return block.type === "text";
}

// --- Plugin ---

const llmProxyPlugin = {
  id: "llm-proxy",
  name: "LLM Proxy",
  description: "Lightweight OpenAI-compatible LLM proxy without agent overhead",
  configSchema: emptyPluginConfigSchema(),
  register(api: { runtime: { config: { loadConfig: () => Record<string, unknown> } }; registerHttpHandler: (handler: (req: IncomingMessage, res: ServerResponse) => Promise<boolean>) => void }) {
    api.registerHttpHandler(async (req, res) => {
      const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
      if (url.pathname !== "/llm/v1/chat/completions") return false;

      if (req.method !== "POST") {
        sendJson(res, 405, { error: { message: "Method Not Allowed", type: "invalid_request_error" } });
        return true;
      }

      // Auth: resolve gateway token/password and compare with request bearer token.
      const cfg = api.runtime.config.loadConfig() as Record<string, unknown>;
      const gatewayConfig = cfg.gateway as Record<string, unknown> | undefined;
      const authConfig = gatewayConfig?.auth as Record<string, unknown> | undefined;
      const resolved = resolveGatewayAuth({ authConfig: authConfig as Parameters<typeof resolveGatewayAuth>[0]["authConfig"] });
      const expectedSecret = resolved.token ?? resolved.password;
      const reqToken = getBearerToken(req);

      if (expectedSecret && (!reqToken || reqToken !== expectedSecret)) {
        sendJson(res, 401, { error: { message: "Unauthorized", type: "unauthorized" } });
        return true;
      }

      // Parse body.
      let body: Record<string, unknown>;
      try {
        const raw = await readBody(req, MAX_BODY_BYTES);
        body = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        sendJson(res, 400, { error: { message: "Invalid JSON body", type: "invalid_request_error" } });
        return true;
      }

      if (body.stream) {
        sendJson(res, 400, { error: { message: "Streaming is not supported by llm-proxy. Use the agent endpoint /v1/chat/completions for streaming.", type: "invalid_request_error" } });
        return true;
      }

      // Extract messages.
      const messages = Array.isArray(body.messages) ? (body.messages as ChatMessage[]) : [];
      const systemParts: string[] = [];
      const contextMessages: Array<{ role: "user" | "assistant"; content: string; timestamp: number }> = [];

      for (const msg of messages) {
        const role = typeof msg.role === "string" ? msg.role.trim() : "";
        const content = extractTextContent(msg.content).trim();
        if (!role || !content) continue;
        if (role === "system" || role === "developer") {
          systemParts.push(content);
          continue;
        }
        if (role === "user" || role === "assistant") {
          contextMessages.push({ role, content, timestamp: Date.now() });
        }
      }

      if (contextMessages.length === 0) {
        sendJson(res, 400, { error: { message: "No user message in `messages`.", type: "invalid_request_error" } });
        return true;
      }

      // Prepend system prompt to the first user message if present.
      if (systemParts.length > 0 && contextMessages.length > 0) {
        const systemPrompt = systemParts.join("\n\n");
        const first = contextMessages[0]!;
        if (first.role === "user") {
          first.content = `${systemPrompt}\n\n${first.content}`;
        } else {
          contextMessages.unshift({ role: "user", content: systemPrompt, timestamp: Date.now() });
        }
      }

      // Resolve model.
      const defaultRef = resolveConfiguredModelRef({
        cfg: cfg as Parameters<typeof resolveConfiguredModelRef>[0]["cfg"],
        defaultProvider: DEFAULT_PROVIDER,
        defaultModel: DEFAULT_MODEL,
      });
      const requestModel = typeof body.model === "string" ? body.model : undefined;
      const { provider, model: modelId } = parseModelRef(requestModel, defaultRef.provider, defaultRef.model);
      const agentDir = resolveAgentDir(
        cfg as Parameters<typeof resolveAgentDir>[0],
        "default",
      );
      const resolved2 = resolveModel(provider, modelId, agentDir, cfg as Parameters<typeof resolveModel>[3]);
      if (!resolved2.model) {
        sendJson(res, 400, { error: { message: resolved2.error ?? `Unknown model: ${provider}/${modelId}`, type: "invalid_request_error" } });
        return true;
      }

      // Get API key.
      const apiKeyResult = await getApiKeyForModel({
        model: resolved2.model,
        cfg: cfg as Parameters<typeof getApiKeyForModel>[0]["cfg"],
        agentDir,
      });
      const apiKey = requireApiKey(apiKeyResult, provider);

      // Call LLM.
      try {
        const llmResult = await completeSimple(
          resolved2.model,
          { messages: contextMessages },
          {
            apiKey,
            maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : 4096,
            ...(typeof body.temperature === "number" ? { temperature: body.temperature } : {}),
          },
        );

        const content = llmResult.content
          .filter(isTextBlock)
          .map((block) => block.text.trim())
          .filter(Boolean)
          .join("\n\n");

        const usage = llmResult.usage ?? { input: 0, output: 0 };

        sendJson(res, 200, {
          id: `chatcmpl_${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: `${provider}/${modelId}`,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: usage.input ?? 0,
            completion_tokens: usage.output ?? 0,
            total_tokens: (usage.input ?? 0) + (usage.output ?? 0),
          },
        });
      } catch (err) {
        sendJson(res, 500, {
          error: { message: String(err), type: "api_error" },
        });
      }

      return true;
    });
  },
};

export default llmProxyPlugin;
