# QuixiAI VectorGraph

A minimal, batteries-included PostgreSQL stack that pairs Apache AGE (graph) with pgvector. Spin it up with Docker, hit a couple of Python helpers, and you have graph + vector storage in one place.

## 60-second start
1. Copy env: `cp .env.example .env`
2. Bring up services: `vectorgraph up` (uses Docker; falls back to `docker compose up -d` with cached stack files)
3. Run tests: `pytest -q`
4. Tinker in Python (see below), or connect to Postgres on `localhost:5432`.

Install options:
- `pipx install .` (recommended for CLI) or `pip install .` in a venv.
- CLI commands: `vectorgraph up`, `vectorgraph down`, `vectorgraph logs -f`, `vectorgraph ps`.

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
More complete example: `python examples/demo.py` (creates types/entities/relationships, vectors, queries, and cleans up).

## Files
- `db.py` — public async API for graph + vector helpers (AGE + pgvector).
- `graph.py` / `vector.py` — thin wrappers if you prefer to import per-domain.
- `schema.sql` — enables extensions and embeds the TEI-friendly `get_embedding` function.
- `Dockerfile` — Postgres 16 image with AGE, pgvector, pgsql-http.
- `docker-compose.yml` — Postgres + HuggingFace TEI (embedding service).
- `tests/` — async end-to-end tests for graph and vector paths.
- `pyproject.toml` — package metadata (dependencies via pip/uv/pdm) and CLI entrypoint.
- `vectorgraph/stack/` — packaged `docker-compose.yml`, `Dockerfile`, `schema.sql` used by the CLI.

## Environment
`POSTGRES_*` are read by both the Python helpers and the Docker stack. The CLI prefers `.env` in your current directory; if absent, it will use one in its cache directory (`~/.cache/vectorgraph/stack/`) if present. The embedding container sits on a private Docker network (no host port) and is reachable from Postgres at `http://embeddings:80`.

## Typical flow
- `docker compose up -d`
- run Python code using the helpers
- `pytest -q` to sanity check
- `docker compose down -v` when done (drops volumes)

## Notes
- Vectors are fixed at 768-dim; the TEI model (`unsloth/embeddinggemma-300m`) matches that.
- Each call to `create_db()` makes a dedicated AGE graph + vector table keyed by UUID to keep tests isolated.
```
