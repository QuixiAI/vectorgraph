import os
import asyncpg
import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

# Allow tests to import project modules without installing a package
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@pytest_asyncio.fixture(scope="function")
async def db_pool():
    pool = await asyncpg.create_pool(
        DB_DSN,
        ssl=False,
        min_size=2,
        max_size=20,
        command_timeout=60.0,
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture(autouse=True)
async def setup_db(db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
    yield
