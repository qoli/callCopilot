#!/usr/bin/env node
import { createServer } from "node:http";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const args = parseArgs(process.argv.slice(2));
const apiKey = process.env.DEEPSEEK_API_KEY;
if (!apiKey) {
  throw new Error("DEEPSEEK_API_KEY is required");
}

const packageRoot = process.env.DEEPSEEK_THINKING_FIX_PACKAGE_ROOT;
if (!packageRoot) {
  throw new Error("DEEPSEEK_THINKING_FIX_PACKAGE_ROOT is required");
}

const api = await import(pathToFileURL(path.join(packageRoot, "dist", "api.js")));
const streamApi = await import(pathToFileURL(path.join(packageRoot, "dist", "stream.js")));

const {
  ThinkingCache,
  assistantTurnFingerprint,
  reinjectThinkingBlocks,
  fillThinkingPlaceholder,
  extractThinkingFromResponse,
} = api;
const { interceptStreamForThinking } = streamApi;

const upstreamBaseUrl = (args["upstream-base-url"] ?? "https://api.deepseek.com/anthropic").replace(/\/+$/, "");
const host = args.host ?? "127.0.0.1";
const port = Number(args.port ?? "0");
const readyFile = args["ready-file"];
const mode = args.mode ?? process.env.DEEPSEEK_THINKING_MODE ?? "preserve";
const debug = args.debug === "1" || process.env.DEEPSEEK_THINKING_PROXY_DEBUG === "1";
const thinkingBudget = Number(args["thinking-budget"] ?? process.env.DEEPSEEK_THINKING_BUDGET ?? "8000");
const placeholderMode = args["placeholder-mode"] ?? process.env.DEEPSEEK_THINKING_PLACEHOLDER_MODE ?? "fallback";
const cache = new ThinkingCache(Number(process.env.DEEPSEEK_THINKING_CACHE_TTL_MS ?? "1800000"));

const server = createServer(async (req, res) => {
  try {
    await handleRequest(req, res);
  } catch (error) {
    console.error("[deepseek-thinking-proxy] error", error);
    if (!res.headersSent) {
      res.writeHead(502, { "content-type": "application/json" });
    }
    res.end(JSON.stringify({ error: { message: String(error?.message ?? error) } }));
  }
});

server.listen(port, host, () => {
  const address = server.address();
  const actualPort = typeof address === "object" && address ? address.port : port;
  const baseUrl = `http://${host}:${actualPort}`;
  if (readyFile) {
    fs.writeFileSync(readyFile, `${baseUrl}\n`);
  }
  console.error(`[deepseek-thinking-proxy] listening on ${baseUrl} mode=${mode}`);
});

async function handleRequest(req, res) {
  if (!req.url) {
    res.writeHead(400);
    res.end("missing URL");
    return;
  }

  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }

  const rawBody = Buffer.concat(chunks).toString("utf8");
  const headers = forwardHeaders(req.headers);
  let bodyText = rawBody;
  let requestBody = null;

  if (req.method !== "GET" && rawBody && isAnthropicMessagesPath(req.url)) {
    requestBody = JSON.parse(rawBody);
    patchAnthropicBody(requestBody);
    bodyText = JSON.stringify(requestBody);
    headers.set("content-length", Buffer.byteLength(bodyText).toString());
  }

  const upstreamResponse = await fetch(`${upstreamBaseUrl}${req.url}`, {
    method: req.method,
    headers,
    body: req.method === "GET" || req.method === "HEAD" ? undefined : bodyText,
  });

  res.writeHead(upstreamResponse.status, upstreamResponse.statusText, responseHeadersForNode(upstreamResponse.headers));

  if (!requestBody || !upstreamResponse.ok) {
    await pipeResponse(upstreamResponse, res);
    return;
  }

  const contentType = upstreamResponse.headers.get("content-type") ?? "";
  const isStream = contentType.includes("text/event-stream") || requestBody.stream === true;
  const nextFingerprint = assistantTurnFingerprint(
    [...(requestBody.messages ?? []), { role: "assistant", content: [] }],
    requestBody.messages?.length ?? 0,
    requestBody.model,
  );

  if (isStream) {
    const intercepted = interceptStreamForThinking(upstreamResponse, (blocks) => {
      if (blocks?.length) {
        cache.set(nextFingerprint, blocks);
        log(`cached ${blocks.length} thinking block(s) from stream`);
      }
    });
    await pipeResponse(intercepted, res);
    return;
  }

  const text = await upstreamResponse.text();
  try {
    const blocks = extractThinkingFromResponse(JSON.parse(text));
    if (blocks?.length) {
      cache.set(nextFingerprint, blocks);
      log(`cached ${blocks.length} thinking block(s) from json`);
    }
  } catch {
    // The upstream body is still returned unchanged.
  }
  res.end(text);
}

function patchAnthropicBody(body) {
  if (!body || !Array.isArray(body.messages)) {
    return;
  }

  if (mode === "disabled") {
    body.thinking = { type: "disabled" };
    return;
  }

  if (body.thinking == null) {
    body.thinking = { type: "enabled", budget_tokens: thinkingBudget };
  }

  const mutated = reinjectThinkingBlocks(body, (fingerprint) => cache.get(fingerprint));
  if (mutated) {
    log("re-injected cached thinking blocks");
  }

  if (placeholderMode !== "off") {
    const filled = fillThinkingPlaceholder(body, {
      mode: placeholderMode,
      text: "(thinking omitted)",
      signaturePolicy: "empty",
    });
    if (filled) {
      log(`filled ${filled} placeholder thinking block(s)`);
    }
  }
}

function isAnthropicMessagesPath(url) {
  const pathname = url.split("?", 1)[0].toLowerCase();
  return pathname.endsWith("/v1/messages") || pathname.endsWith("/messages");
}

function forwardHeaders(inputHeaders) {
  const headers = new Headers();
  const skip = new Set(["connection", "content-length", "host", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"]);
  for (const [name, value] of Object.entries(inputHeaders)) {
    if (skip.has(name.toLowerCase()) || value == null) {
      continue;
    }
    headers.set(name, Array.isArray(value) ? value.join(", ") : value);
  }
  if (!headers.has("x-api-key") && !headers.has("authorization")) {
    headers.set("x-api-key", apiKey);
  }
  headers.set("content-type", headers.get("content-type") ?? "application/json");
  return headers;
}

function responseHeadersForNode(headers) {
  const out = {};
  const skip = new Set(["connection", "content-encoding", "content-length", "keep-alive", "transfer-encoding"]);
  for (const [name, value] of headers.entries()) {
    if (!skip.has(name.toLowerCase())) {
      out[name] = value;
    }
  }
  out.connection = "close";
  return out;
}

async function pipeResponse(response, res) {
  if (!response.body) {
    res.end();
    return;
  }
  await new Promise((resolve, reject) => {
    Readable.fromWeb(response.body).on("error", reject).on("end", resolve).pipe(res);
  });
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      continue;
    }
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = "1";
    } else {
      parsed[key] = next;
      i++;
    }
  }
  return parsed;
}

function log(message) {
  if (debug) {
    console.error(`[deepseek-thinking-proxy] ${message}`);
  }
}
