// Resource bounds shared by public JSON routes. A Content-Length check alone
// cannot bound chunked bodies; count bytes while reading before JSON parsing.
export const MAX_REQUEST_BYTES = 192 * 1024;
export const MAX_PROMPT_CHARS = 40_000;
export const MAX_TRIPLES = 256;
export const MAX_TRIPLE_PART_CHARS = 512;
export const PROVIDER_TIMEOUT_MS = 30_000;

export class RequestError extends Error {
  status: number;
  constructor(message: string, status = 400) { super(message); this.status = status; }
}

// The headers every route's own json() helper sends: mutating endpoints are
// no-store (API_REFERENCE §2.6), request errors included.
const ERROR_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
export function errorResponse(error: unknown): Response {
  const known = error instanceof RequestError;
  return new Response(JSON.stringify({ error: known ? error.message : "invalid JSON body" }), { status: known ? error.status : 400, headers: ERROR_HEADERS });
}

export async function readBoundedJSON(
  message: Request | Response,
  maxBytes = MAX_REQUEST_BYTES,
  timeoutMs = 10_000,
): Promise<unknown> {
  const length = message.headers.get("content-length");
  if (length && Number(length) > maxBytes) throw new RequestError(`JSON body exceeds ${maxBytes} bytes`, 413);
  const reader = message.body?.getReader();
  if (!reader) throw new RequestError("missing JSON body");
  // A fixed buffer and one deadline race avoid per-chunk retained promises
  // or arrays when an input arrives in many tiny chunks.
  const bytes = new Uint8Array(maxBytes);
  let size = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const read = async () => {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (size + value.byteLength > maxBytes) {
        void reader.cancel().catch(() => {});
        throw new RequestError(`JSON body exceeds ${maxBytes} bytes`, 413);
      }
      bytes.set(value, size);
      size += value.byteLength;
    }
    return JSON.parse(new TextDecoder("utf-8", { fatal: true, ignoreBOM: false }).decode(bytes.subarray(0, size)));
  };
  try {
    const deadline = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        reject(new RequestError("JSON body read timed out", 408));
        void reader.cancel().catch(() => {});
      }, timeoutMs);
    });
    return await Promise.race([read(), deadline]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    reader.releaseLock();
  }

}

export function requireObject(value: unknown, name = "body"): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new RequestError(`${name} must be an object`);
}

// A lone UTF-16 surrogate has no UTF-8 encoding: canonicalize() and
// encodeURIComponent() throw on it, and Python's encoder refuses it too. It is
// bad client input, so a triple, a source_chain claim or source_uri, a qid
// term's text and a completion prompt that holds one are refused with 400
// before any hashing, signing, caching, lookup or provider call. (\p{Cs}
// matches only unpaired surrogates under the
// u flag; isWellFormed() is ES2024 and outside tsconfig's ES2022 lib.)
const LONE_SURROGATE = /\p{Cs}/u;
export function isWellFormedText(value: string): boolean {
  return !LONE_SURROGATE.test(value);
}

export function validateTriples(value: unknown): asserts value is Array<[string, string, string]> {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_TRIPLES) throw new RequestError(`triples must contain 1 to ${MAX_TRIPLES} items`);
  let characters = 0;
  for (const triple of value) {
    if (!Array.isArray(triple) || triple.length !== 3 || triple.some(part => typeof part !== "string" || !part.trim() || part.length > MAX_TRIPLE_PART_CHARS)) {
      throw new RequestError(`each triple must contain three nonempty strings of at most ${MAX_TRIPLE_PART_CHARS} characters`);
    }
    if (!triple.every(isWellFormedText)) throw new RequestError("triple text must be well-formed Unicode (no lone surrogates)");
    characters += triple.reduce((n: number, part: string) => n + part.length, 0);
  }
  if (characters > MAX_PROMPT_CHARS) throw new RequestError(`triple text exceeds ${MAX_PROMPT_CHARS} characters`, 413);
}

export function validateSliders(value: unknown): void {
  requireObject(value, "sliders");
  for (const axis of ["density", "length", "formality", "audience", "perspective"]) {
    if (typeof value[axis] !== "number" || !Number.isFinite(value[axis]) || value[axis] < 0 || value[axis] > 1) {
      throw new RequestError(`${axis} must be a finite number in [0, 1]`);
    }
  }
}

// The timeout covers headers AND consumption of the response body. Aborting
// local work cannot guarantee that an upstream provider stops billable work.
export async function providerJSON(url: string, init: RequestInit, timeoutMs = PROVIDER_TIMEOUT_MS): Promise<unknown> {
  const deadline = Date.now() + timeoutMs;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      void response.body?.cancel().catch(() => {});
      throw new Error(`provider returned HTTP ${response.status}`);
    }
    return await readBoundedJSON(response, 256 * 1024, Math.max(1, deadline - Date.now()));
  } catch (error) {
    if (controller.signal.aborted) throw new Error("provider request timed out");
    throw error;
  } finally { if (timer !== undefined) clearTimeout(timer); }
}
