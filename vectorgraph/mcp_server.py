"""Minimal MCP server exposing VectorGraph helpers."""
import asyncio
import json
from importlib.metadata import version, PackageNotFoundError
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, Tool, ToolsCapability, TextContent

from vectorgraph import (
    create_db,
    delete_db,
    graph_create_entity,
    graph_create_relationship,
    graph_get_entity,
    graph_neighbors,
    graph_similarity,
    vector_add,
    vector_batch_add,
    vector_batch_delete,
    vector_get,
    vector_nearest_neighbors,
    vector_query_by_id,
    cross_join_query,
)

# Honor .env when present so the server connects to the right Postgres instance.
load_dotenv()

server = Server("vectorgraph-mcp")


def _tool(name: str, description: str, schema: Dict[str, Any]) -> Tool:
    return Tool(name=name, description=description, inputSchema=schema)


TOOLS: List[Tool] = [
    _tool(
        "create_db",
        "Create a new graph+vector namespace and return its id.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    _tool(
        "delete_db",
        "Drop a graph+vector namespace created with create_db.",
        {
            "type": "object",
            "properties": {"db_id": {"type": "string"}},
            "required": ["db_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "graph_create_entity",
        "Create an Entity node.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
            },
            "required": ["db_id", "id", "name"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "graph_create_relationship",
        "Create a relationship between two entities.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "relation": {"type": "string"},
                "properties": {"type": "object"},
            },
            "required": ["db_id", "source_id", "target_id", "relation"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "graph_get_entity",
        "Fetch an Entity node by id.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "id": {"type": "string"},
            },
            "required": ["db_id", "id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "graph_neighbors",
        "List neighbors for an entity.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "entity_id": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                "relation": {"type": ["string", "null"]},
            },
            "required": ["db_id", "entity_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "graph_similarity",
        "Compute similarity between two entities.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "entity_a": {"type": "string"},
                "entity_b": {"type": "string"},
                "method": {"type": "string", "enum": ["neighbors"], "default": "neighbors"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            },
            "required": ["db_id", "entity_a", "entity_b"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_add",
        "Insert a vector row with metadata.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "id": {"type": "string"},
                "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
                "metadata": {"type": "object"},
            },
            "required": ["db_id", "id", "vector"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_get",
        "Fetch a vector row by id.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "id": {"type": "string"},
                "include_vector": {"type": "boolean", "default": False},
            },
            "required": ["db_id", "id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_nearest_neighbors",
        "Query nearest neighbors by vector.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
                "k": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10},
                "metadata_filter": {"type": "object"},
                "include_vector": {"type": "boolean", "default": False},
            },
            "required": ["db_id", "vector"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_query_by_id",
        "Query neighbors using an existing row's vector.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "id": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10},
                "metadata_filter": {"type": "object"},
                "include_vector": {"type": "boolean", "default": False},
            },
            "required": ["db_id", "id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "cross_join_query",
        "Vector -> SQL -> graph flow: vector search, optional SQL filter on a table, optional graph neighbors.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
                "source_id": {"type": "string", "description": "Use this row's vector if 'vector' is not provided."},
                "k": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10},
                "table": {"type": "string", "description": "Relational table to filter/join by id."},
                "where": {"type": "object", "description": "Equality predicates for the table (column: value)."},
                "include_neighbors": {"type": "boolean", "default": True},
                "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                "relation": {"type": ["string", "null"]},
                "include_vector": {"type": "boolean", "default": False},
            },
            "required": ["db_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_batch_add",
        "Insert or upsert multiple vectors.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "vector": {"type": "array", "items": {"type": "number"}, "minItems": 768, "maxItems": 768},
                            "metadata": {"type": "object"},
                        },
                        "required": ["vector"],
                        "additionalProperties": False,
                    },
                },
                "upsert": {"type": "boolean", "default": False},
            },
            "required": ["db_id", "items"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "vector_batch_delete",
        "Delete multiple vectors by id.",
        {
            "type": "object",
            "properties": {
                "db_id": {"type": "string"},
                "ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["db_id", "ids"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "batch",
        "Run multiple MCP tool calls sequentially.",
        {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object", "default": {}},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                "continue_on_error": {"type": "boolean", "default": False},
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
    ),
]


def _require(args: Dict[str, Any], key: str, tool: str) -> Any:
    if key not in args:
        raise ValueError(f"Missing required argument '{key}' for tool '{tool}'")
    return args[key]


async def _dispatch_tool(name: str, args: Dict[str, Any]) -> Any:
    if name == "create_db":
        db_id = await create_db()
        return {"db_id": db_id}
    if name == "delete_db":
        db_id = _require(args, "db_id", name)
        await delete_db(db_id)
        return {"deleted": db_id}
    if name == "graph_create_entity":
        return await graph_create_entity(
            _require(args, "db_id", name),
            _require(args, "id", name),
            _require(args, "name", name),
            args.get("description"),
        )
    if name == "graph_create_relationship":
        return await graph_create_relationship(
            _require(args, "db_id", name),
            _require(args, "source_id", name),
            _require(args, "target_id", name),
            _require(args, "relation", name),
            **(args.get("properties") or {}),
        )
    if name == "graph_get_entity":
        return await graph_get_entity(_require(args, "db_id", name), _require(args, "id", name))
    if name == "graph_neighbors":
        return await graph_neighbors(
            _require(args, "db_id", name),
            _require(args, "entity_id", name),
            depth=args.get("depth", 1),
            relation=args.get("relation"),
        )
    if name == "graph_similarity":
        return await graph_similarity(
            _require(args, "db_id", name),
            _require(args, "entity_a", name),
            _require(args, "entity_b", name),
            method=args.get("method", "neighbors"),
            depth=args.get("depth", 2),
        )
    if name == "vector_add":
        return await vector_add(
            _require(args, "db_id", name),
            _require(args, "id", name),
            _require(args, "vector", name),
            args.get("metadata"),
        )
    if name == "vector_get":
        return await vector_get(
            _require(args, "db_id", name),
            _require(args, "id", name),
            include_vector=args.get("include_vector", False),
        )
    if name == "vector_nearest_neighbors":
        return await vector_nearest_neighbors(
            _require(args, "db_id", name),
            _require(args, "vector", name),
            k=args.get("k", 10),
            metadata_filter=args.get("metadata_filter"),
            include_vector=args.get("include_vector", False),
        )
    if name == "vector_query_by_id":
        return await vector_query_by_id(
            _require(args, "db_id", name),
            _require(args, "id", name),
            k=args.get("k", 10),
            metadata_filter=args.get("metadata_filter"),
            include_vector=args.get("include_vector", False),
        )
    if name == "cross_join_query":
        return await cross_join_query(
            _require(args, "db_id", name),
            vector=args.get("vector"),
            source_id=args.get("source_id"),
            k=args.get("k", 10),
            table=args.get("table"),
            where=args.get("where"),
            include_neighbors=args.get("include_neighbors", True),
            depth=args.get("depth", 1),
            relation=args.get("relation"),
            include_vector=args.get("include_vector", False),
        )
    if name == "vector_batch_add":
        return await vector_batch_add(
            _require(args, "db_id", name),
            _require(args, "items", name),
            upsert=args.get("upsert", False),
        )
    if name == "vector_batch_delete":
        return await vector_batch_delete(
            _require(args, "db_id", name),
            _require(args, "ids", name),
        )
    if name == "batch":
        operations = _require(args, "operations", name) or []
        continue_on_error = args.get("continue_on_error", False)
        results = []
        for op in operations:
            op_name = op.get("name") if isinstance(op, dict) else None
            op_args = op.get("arguments") if isinstance(op, dict) else {}
            try:
                if not isinstance(op_name, str):
                    raise ValueError("operation.name must be a string")
                result = await _dispatch_tool(op_name, op_args or {})
                results.append({"name": op_name, "result": result})
            except Exception as exc:
                error_payload = {"name": op_name or "<unknown>", "error": str(exc)}
                results.append(error_payload)
                if not continue_on_error:
                    raise
        return results
    raise ValueError(f"Unknown tool '{name}'")


@server.list_tools()
async def list_tools() -> List[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):
    try:
        result = await _dispatch_tool(name, arguments or {})
        text = json.dumps(result, indent=2, sort_keys=True, default=str)
    except Exception as exc:  # pragma: no cover - surfaced to MCP client
        text = f"Error: {exc}"
    return [TextContent(type="text", text=text)]


async def main():
    # Run MCP server over stdio
    try:
        server_version = version("vectorgraph")
    except PackageNotFoundError:  # pragma: no cover - local dev
        server_version = "dev"

    init_opts = InitializationOptions(
        server_name="vectorgraph-mcp",
        server_version=server_version,
        capabilities=ServerCapabilities(tools=ToolsCapability()),
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_opts)


if __name__ == "__main__":
    asyncio.run(main())
