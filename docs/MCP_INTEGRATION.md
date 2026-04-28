# MCP integration — calling SUM from MCP-aware LLM clients

SUM ships a Model Context Protocol (MCP) server that exposes its primary verbs (`extract`, `attest`, `verify`, `inspect`, `schema`) as MCP tools. Any MCP-aware client — Claude Desktop, Claude Code, Cursor, Continue, custom agents built on the MCP Python or TypeScript SDKs — can drive SUM directly without shelling out to the `sum` CLI or hitting the hosted Worker API.

This is the integration surface for **systems calling SUM** — the most common deployment shape.

## Install

```bash
pip install 'sum-engine[mcp,sieve]'   # MCP server + offline sieve extractor
# or
pip install 'sum-engine[mcp,llm]'     # MCP server + OpenAI structured-output extractor
# or
pip install 'sum-engine[all]'         # everything (sieve + llm + receipt-verify + mcp + dev)
```

After install, two console scripts land on PATH:

| binary | purpose |
|---|---|
| `sum`     | the existing CLI (`sum attest`, `sum verify`, `sum ledger ...`) |
| `sum-mcp` | the MCP server, stdio transport (newline-delimited JSON-RPC 2.0) |

Equivalent invocation without installing the script: `python -m sum_engine_internal.mcp_server`.

## Wire format

MCP's stdio transport: the client spawns `sum-mcp` as a subprocess, writes JSON-RPC 2.0 requests to stdin, reads responses from stdout. This is the standard MCP wire for **local** LLM clients. (Remote SSE / HTTP transports are not enabled in v1; they are a follow-on once authentication semantics are designed — `sum-mcp` over the network is a different threat model than `sum-mcp` on the same host.)

The five tools the server registers:

### `extract`

```
extract(text: str, extractor: "auto" | "sieve" | "llm" = "auto") -> {
  triples: [[s, p, o], ...],
  extractor: <chosen>,
  count: <int>,
} | { error: <message> }
```

Pulls (subject, predicate, object) triples out of natural-language prose. Fast, side-effect-free; no canonical bundle, no signing, no state integer. The fastest tool for "what does this text mean to SUM."

### `attest`

```
attest(
  text: str,
  extractor: "auto" | "sieve" | "llm" = "auto",
  branch: str = "main",
  title: str | None = None,
  signing_key: str | None = None,
) -> {
  bundle: <CanonicalBundle JSON>,
  axioms: <int>,
  source_uri: "sha256:<hex>",
  extractor: <chosen>,
} | { error: <message> }
```

The full pipeline: extract triples, encode them into the Gödel state integer, generate the canonical tome, sign with Ed25519, optionally HMAC. The bundle this returns is **byte-identical** to what `sum attest` produces from the CLI — verifiable by every existing SUM verifier (Python, Node `standalone_verifier`, browser `single_file_demo`).

### `verify`

```
verify(bundle: dict, signing_key: str | None = None, strict: bool = False) -> {
  ok: <bool>,
  axioms: <int>,
  state_integer_digits: <int>,
  branch: <str>,
  bundle_version: <str>,
  signatures: { ed25519: "valid" | "invalid" | "absent",
                hmac: "valid" | "invalid" | "absent" | "skipped" },
  errors: [<reason>, ...],
}
```

Six-step verification: schema-version gate → prime-scheme gate → Ed25519 signature → HMAC signature → canonical-tome → state-integer reconstruction → axiom-count match. Returns a structured dict, never raises on malformed input — "verifier said no" stays distinct from "verifier crashed."

### `inspect`

```
inspect(bundle: dict) -> {
  branch, title, axiom_count, bundle_version,
  canonical_format_version, prime_scheme,
  state_integer_digits, tome_lines,
  signatures_present: { ed25519: <bool>, hmac: <bool> },
  sum_cli: <sidecar dict or None>,
}
```

Read-only summary — does **not** verify, does **not** reconstruct state. The "what's in this bundle" view an LLM agent calls before deciding whether to run the more-expensive `verify`.

### `schema`

```
schema(name: "list" | "sum.canonical_bundle.v1" | "sum.render_receipt.v1" | "sum.merkle_inclusion.v1" = "list")
  -> { schemas: [...] }                 // for name="list"
   | { schema, version, fields, spec }  // for a named schema
   | { error, known: [...] }            // for an unknown name
```

Returns the field catalogue for the canonical SUM artifacts. The wire-spec sources of truth remain `docs/RENDER_RECEIPT_FORMAT.md` and `docs/MERKLE_SIDECAR_FORMAT.md`; this tool gives an in-band, programmatically-readable summary so an agent doesn't have to fetch and parse markdown to know what fields to expect.

## Client configuration

### Claude Desktop / Claude Code

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the Claude Code MCP settings:

```json
{
  "mcpServers": {
    "sum": {
      "command": "sum-mcp"
    }
  }
}
```

If `sum-mcp` is not on PATH (e.g. virtualenv-isolated install), use the absolute path or the `python -m` form:

```json
{
  "mcpServers": {
    "sum": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "sum_engine_internal.mcp_server"]
    }
  }
}
```

### Cursor / Continue

Same shape; both honour the standard MCP server config block. Restart the editor after editing the config so the MCP host picks up the new server.

### Custom agents

```python
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp import StdioServerParameters

params = StdioServerParameters(command="sum-mcp")
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(
            "attest",
            {"text": "The printing press was invented by Johannes Gutenberg."},
        )
        bundle = result.content[0].text  # JSON-encoded result dict
```

## Trust model

The MCP server is a façade over `sum_engine_internal` and `sum_cli.main`. **No new cryptography, no new canonical codec.** A bundle attested via MCP:

- Produces the same canonical bytes the CLI does.
- Verifies via the same Python / Node / browser verifiers without modification.
- Inherits the trust scope documented in `docs/RENDER_RECEIPT_FORMAT.md` §"Trust Scope" and `docs/PROOF_BOUNDARY.md` §1.3.1.

What the MCP server **does not** do:

- It does not verify what the user said — `extract` is whatever the chosen extractor produces. The trust boundary is at attest-time signing, same as the CLI.
- It does not run on a remote endpoint. v1 is stdio-only. A remote-MCP variant (SSE / HTTP) is gated on an auth design that does not yet exist.
- It does not write to the Akashic Ledger. Ledger writes happen only via `sum attest --ledger` from the CLI — surfacing that path through MCP requires a separate review of provenance-tracking semantics, since MCP tool calls are intended to be safe-by-default.

## Cross-references

- `docs/PROOF_BOUNDARY.md` §1.3.1 — cross-runtime trust triangle (Python ↔ Node ↔ browser). MCP-attested bundles inherit this.
- `docs/RENDER_RECEIPT_FORMAT.md` — wire spec for the receipt path.
- `Tests/test_mcp_server.py` — registration, roundtrip, and CLI-equivalence coverage.
- MCP Python SDK: <https://github.com/modelcontextprotocol/python-sdk>
- MCP spec: <https://spec.modelcontextprotocol.io>
