# Repository Guidelines

## Project Structure & Modules
- Python package lives in `vectorgraph/`; entrypoints in `vectorgraph/db.py`, `graph.py`, `vector.py`, `sync.py`, and CLI in `cli.py`.
- Packaged stack assets (compose, Dockerfile, schema) are under `vectorgraph/stack/`; demo script sits in `vectorgraph/examples/demo.py`.
- Tests are under `tests/` and exercise async flows against the Docker stack.
- Root artifacts: `Dockerfile`, `docker-compose.yml`, `schema.sql`, `pyproject.toml`, and `README.md` for quick usage.

## Build, Test, and Development Commands
- Install editable with dev extras: `pip install -e .[dev]`.
- Bring up the stack (Postgres + AGE + pgvector + embeddings): `vectorgraph up`; tear down with `vectorgraph down`; stream logs with `vectorgraph logs -f`.
- Quick demo: `vectorgraph demo` then `python demo.py` (generates a sample script).
- Run tests (async): `pytest -q`; with coverage: `pytest --cov=vectorgraph --cov-report=term-missing`.
- Package build: `python -m build` (requires `build` if not already installed).

## Coding Style & Naming
- Python 3.10+; prefer async APIs (`vectorgraph.db`, `graph`, `vector`) and expose sync shims in `vectorgraph.sync`.
- Formatting: `black` and `isort`; type checks with `mypy`. Run `black . && isort .` before PRs.
- Tests and async helpers use `asyncio_mode=auto` per `pytest.ini`; keep fixtures function-scoped unless needed otherwise.
- Naming: modules and files are snake_case; CLI commands remain kebab-case (`vectorgraph up`).

## Testing Guidelines
- Primary suite is `pytest` with `pytest-asyncio`; aim to keep async tests isolated by creating a fresh DB per case (see helpers in tests).
- Name tests `test_*.py` and async coroutines `async def test_*`.
- Run the stack locally (`vectorgraph up`) before executing integration tests; ensure Docker is available.

## Commit & Pull Request Guidelines
- Follow concise, present-tense commit subjects (see history: e.g., `add batch operations for mcp`).
- PRs should include: brief summary, linked issue (if applicable), and any notes on stack changes or required env vars.
- Add logs or screenshots only when UI/CLI output is relevant; otherwise cite commands used for verification.

## Security & Configuration Tips
- Environment defaults are baked into the CLI; to point at a custom Postgres/AGE instance set `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.
- The embedding service is internal to the compose network; avoid exposing it directly. Use `vectorgraph up` and connect via Postgres.
