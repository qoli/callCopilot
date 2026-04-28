#!/usr/bin/env node
import { createServer } from "node:http";
import { Readable } from "node:stream";
import fs from "node:fs";

const args = parseArgs(process.argv.slice(2));
const apiKey = process.env.OPENAI_COMPAT_API_KEY ?? "";
const upstreamBaseUrl = (args["upstream-base-url"] ?? "http://127.0.0.1:8001/v1").replace(/\/+$/, "");
const host = args.host ?? "127.0.0.1";
const port = Number(args.port ?? "0");
const readyFile = args["ready-file"];

const server = createServer(async (req, res) => {
  try {
    await handleRequest(req, res);
  } catch (error) {
    console.error("[openai-compat-cleanup-proxy] error", error);
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
  console.error(`[openai-compat-cleanup-proxy] listening on ${baseUrl}`);
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

  const bodyText = Buffer.concat(chunks).toString("utf8");
  const upstreamResponse = await fetch(buildUpstreamUrl(upstreamBaseUrl, req.url), {
    method: req.method,
    headers: forwardHeaders(req.headers),
    body: req.method === "GET" || req.method === "HEAD" ? undefined : bodyText,
  });

  res.writeHead(upstreamResponse.status, upstreamResponse.statusText, responseHeadersForNode(upstreamResponse.headers));

  const contentType = upstreamResponse.headers.get("content-type") ?? "";
  if (contentType.includes("text/event-stream")) {
    await pipeOpenAIStreamNormalized(upstreamResponse, res);
    return;
  }

  const text = await upstreamResponse.text();
  try {
    const json = JSON.parse(text);
    normalizeOpenAIResponseJson(json);
    res.end(JSON.stringify(json));
  } catch {
    res.end(text);
  }
}

async function pipeOpenAIStreamNormalized(response, res) {
  if (!response.body) {
    res.end();
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer) {
        res.write(encoder.encode(buffer));
      }
      res.end();
      return;
    }

    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 2);
      res.write(encoder.encode(`${normalizeOpenAIStreamEvent(raw)}\n\n`));
    }
  }
}

function normalizeOpenAIStreamEvent(raw) {
  return raw.split("\n").map((line) => {
    if (!line.startsWith("data:")) {
      return line;
    }
    const prefix = line.match(/^data:\s*/)?.[0] ?? "data: ";
    const payload = line.slice(prefix.length);
    if (!payload || payload === "[DONE]") {
      return line;
    }
    try {
      const json = JSON.parse(payload);
      normalizeOpenAIResponseJson(json);
      return `${prefix}${JSON.stringify(json)}`;
    } catch {
      return line;
    }
  }).join("\n");
}

function normalizeOpenAIResponseJson(json) {
  if (json?.usage === null) {
    delete json.usage;
  }
  const choices = json?.choices;
  if (!Array.isArray(choices)) {
    return;
  }
  for (const choice of choices) {
    if (choice?.logprobs === null) {
      delete choice.logprobs;
    }
    if (choice?.matched_stop === null) {
      delete choice.matched_stop;
    }
    if (choice?.message?.content === null) {
      choice.message.content = "";
    }
    if (choice?.message) {
      deleteNullProperties(choice.message);
    }
    if (choice?.delta) {
      deleteNullProperties(choice.delta);
    }
  }
}

function deleteNullProperties(object) {
  for (const key of Object.keys(object)) {
    if (object[key] === null) {
      delete object[key];
    }
  }
}

function forwardHeaders(inputHeaders) {
  const headers = new Headers();
  const skip = new Set(["connection", "content-length", "host", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"]);
  for (const [name, value] of Object.entries(inputHeaders)) {
    if (skip.has(name.toLowerCase()) || value == null || name.toLowerCase() === "authorization") {
      continue;
    }
    headers.set(name, Array.isArray(value) ? value.join(", ") : value);
  }
  if (apiKey) {
    headers.set("authorization", `Bearer ${apiKey}`);
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

function buildUpstreamUrl(baseUrl, requestUrl) {
  const [rawPath, query = ""] = requestUrl.split("?", 2);
  let path = rawPath;
  if (baseUrl.endsWith("/v1") && path.startsWith("/v1/")) {
    path = path.slice(3);
  }
  return `${baseUrl}${path}${query ? `?${query}` : ""}`;
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
