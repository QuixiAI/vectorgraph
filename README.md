# QuixiAI VectorGraph

A minimal, batteries-included PostgreSQL stack that pairs Apache AGE (graph) with pgvector. Spin it up with Docker, hit a couple of Python helpers, and you have graph + vector storage in one place.

## 60-second start
1. Install: `pip install vectorgraph` (or `pipx install vectorgraph`)
2. Bring up services: `vectorgraph up` (Docker compose stack with graph/vector)
3. Run tests: `pytest -q` (optional if you cloned)
4. Tinker in Python (see below) or run `vectorgraph demo` then `python demo.py`.

Install options:
- `pip install vectorgraph` (or `pipx install vectorgraph` for a global CLI).
- CLI commands: `vectorgraph up`, `vectorgraph down`, `vectorgraph logs -f`, `vectorgraph ps`, `vectorgraph demo`.
  - Prefer async API for apps; sync helpers are available at `vectorgraph.sync` (see async/sync combined demo).

## Python quickstart
```python
import asyncio
from vectorgraph import create_db, delete_db, graph_create_entity, vector_add, vector_nearest_neighbors

async def main():
    db_id = await create_db()
    try:
        await graph_create_entity(db_id, "n1", "Hello", "Graph+Vector")
        await vector_add(db_id, "n1", [0.1]*768, {"label": "hello"})
        neighbors = await vector_nearest_neighbors(db_id, [0.1]*768, k=3)
        print(neighbors)
    finally:
        await delete_db(db_id)

asyncio.run(main())
```
Combined example: `python examples/demo.py` (async flow) and `python examples/demo.py --sync` (sync via `vectorgraph.sync`).

## Use as a library
Install into your app (no CLI needed if you already run Postgres/AGE/pgvector):
```
pip install vectorgraph
```
Minimal usage (sync helpers):
```python
from vectorgraph import sync as vg

db_id = vg.create_db()
vg.vector_add(db_id, "id1", [0.1]*768, {"tag": "demo"})
print(vg.vector_nearest_neighbors(db_id, [0.1]*768, k=1))
vg.delete_db(db_id)
```
Env vars respected by the helpers: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`. If you’re pointing at an existing stack, set these to your running Postgres/AGE instance. Defaults match the bundled compose stack: `POSTGRES_USER=vg_user`, `POSTGRES_PASSWORD=vg_password`, `POSTGRES_DB=vg_db`, `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=5432`.

## Clone the repo (optional)
If you want the source and tests locally:
- Clone: `git clone https://github.com/QuixiAI/vectorgraph.git && cd vectorgraph`
- Install editable: `pip install -e .`
- Run tests: `pytest -q`

## Files
- `db.py` — public async API for graph + vector helpers (AGE + pgvector).
- `graph.py` / `vector.py` — thin wrappers if you prefer to import per-domain.
- `schema.sql` — enables extensions and embeds the TEI-friendly `get_embedding` function.
- `Dockerfile` — Postgres 16 image with AGE, pgvector, pgsql-http.
- `docker-compose.yml` — Postgres + HuggingFace TEI (embedding service).
- `tests/` — async end-to-end tests for graph and vector paths.
- `pyproject.toml` — package metadata (dependencies via pip/uv/pdm) and CLI entrypoint.
- `vectorgraph/stack/` — packaged `docker-compose.yml`, `Dockerfile`, `schema.sql` used by the CLI.
- `vectorgraph-mcp.py` — MCP stdio server exposing VectorGraph tools (for Claude Desktop or other MCP clients).

## Environment
Defaults are baked into the stack; you normally don’t need to touch `.env`. If a `.env` exists in your project root, `vectorgraph up` will copy it into its cache and use it; otherwise it uses the packaged defaults. The embedding container sits on a private Docker network (no host port) and is reachable from Postgres at `http://embeddings:80`.

## Typical flow
- `vectorgraph up`
- run Python code using the helpers (or `vectorgraph demo` then `python demo.py`)
- `pytest -q` to sanity check
- `vectorgraph down` when done

## MCP server (Claude Desktop)
- Ensure the stack is running (`vectorgraph up`) and your env vars point at it if customized.
- Start the server: `vectorgraph mcp` (stdio MCP server; alternatively `python -m vectorgraph.mcp_server`).
- Configure Claude Desktop to point at this server; exposed tools include `create_db`, `delete_db`, graph helpers (`graph_create_entity`, `graph_get_entity`, `graph_create_relationship`, `graph_neighbors`, `graph_similarity`), and vector helpers (`vector_add`, `vector_get`, `vector_nearest_neighbors`, `vector_query_by_id`, `vector_batch_add`, `vector_batch_delete`).
- Example Claude Desktop snippet:
  - Create an MCP server entry named `vectorgraph` with command `vectorgraph mcp` (no args). Leave env empty unless you need custom `POSTGRES_*`.
  - Ask Claude: “Use vectorgraph to create a DB, add an entity id='n1' name='Hello', add a vector [0.1]*768 with metadata {'tag':'demo'}, then fetch nearest neighbors for [0.1]*768.” Claude will call `create_db`, `graph_create_entity`, `vector_add`, and `vector_nearest_neighbors` through the MCP tools.
- Example JSON config snippet for Claude Desktop:
  ```json
  {
    "mcpServers": {
      "vectorgraph": {
        "command": "vectorgraph",
        "args": ["mcp"],
        "env": {
          "POSTGRES_HOST": "127.0.0.1",
          "POSTGRES_PORT": "5432",
          "POSTGRES_USER": "vg_user",
          "POSTGRES_PASSWORD": "vg_password",
          "POSTGRES_DB": "vg_db"
        }
      }
    }
  }
  ```

## Notes
- Vectors are fixed at 768-dim; the TEI model (`unsloth/embeddinggemma-300m`) matches that.
- Each call to `create_db()` makes a dedicated AGE graph + vector table keyed by UUID to keep tests isolated.
