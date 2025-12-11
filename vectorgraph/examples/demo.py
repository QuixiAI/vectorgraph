"""
Example usage of the graph/vector helpers in a realistic flow:
1) create an isolated graph/table
2) add entity types and entities
3) add relationships
4) insert vectors and query nearest neighbors
5) traverse and compute similarity
6) clean up
"""
import argparse
import asyncio
import uuid
from vectorgraph import (
    create_db,
    delete_db,
    graph_assign_type,
    graph_create_entity,
    graph_create_relationship,
    graph_create_type,
    graph_neighbors,
    graph_run_cypher,
    graph_similarity,
    graph_traverse,
    vector_add,
    vector_nearest_neighbors,
    vector_query_by_id,
)
from vectorgraph import sync as vg_sync


def make_vec(v: float) -> list[float]:
    return [v] * 768


async def run_async() -> None:
    db_id = await create_db()
    print(f"Created graph/vector namespace: {db_id}")

    # Domain setup
    person = await graph_create_type(db_id, "Person")
    company = await graph_create_type(db_id, "Company")
    print("Types:", person, company)

    # Note: vector IDs must be valid UUID strings (pg uuid column)
    alice_id = str(uuid.uuid4())
    acme_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())

    alice = await graph_create_entity(db_id, alice_id, "Alice", "Engineer at ACME")
    acme = await graph_create_entity(db_id, acme_id, "ACME", "A rocket company")
    bob = await graph_create_entity(db_id, bob_id, "Bob", "Designer at ACME")

    await graph_assign_type(db_id, alice_id, "Person")
    await graph_assign_type(db_id, bob_id, "Person")
    await graph_assign_type(db_id, acme_id, "Company")

    # Relationships
    await graph_create_relationship(db_id, alice_id, acme_id, "EMPLOYED_AT", since="2022")
    await graph_create_relationship(db_id, bob_id, acme_id, "EMPLOYED_AT", since="2021")
    await graph_create_relationship(db_id, alice_id, bob_id, "COLLABORATES_WITH", project="Design")

    # Vector inserts (e.g., embeddings for descriptions)
    await vector_add(db_id, alice_id, make_vec(0.1), {"name": "Alice", "team": "eng"})
    await vector_add(db_id, bob_id, make_vec(0.11), {"name": "Bob", "team": "design"})
    await vector_add(db_id, acme_id, make_vec(0.5), {"name": "ACME", "team": "org"})

    # Vector similarity: who is closest to Alice?
    neighbors = await vector_nearest_neighbors(db_id, make_vec(0.1), k=2)
    print("Nearest to Alice vector:", neighbors)

    # Metadata-filtered search: find only design team vectors
    design_only = await vector_nearest_neighbors(db_id, make_vec(0.11), k=2, metadata_filter={"team": "design"})
    print("Design-only neighbors:", design_only)

    # Query-by-id to find neighbors of Bob
    bob_neighbors = await vector_query_by_id(db_id, bob_id, k=2)
    print("Neighbors of Bob (by id):", bob_neighbors)

    # Graph traversal: Alice's coworkers within 1 hop
    coworkers = await graph_neighbors(db_id, alice_id, depth=1, relation="COLLABORATES_WITH")
    print("Alice collaborators:", coworkers)

    # Traverse paths up to depth 2 from Alice
    traversal = await graph_traverse(db_id, alice_id, depth=2)
    print("Traversal paths from Alice:", traversal.get("paths"))

    # Graph similarity: overlap of neighbors
    sim = await graph_similarity(db_id, alice_id, bob_id, depth=2)
    print("Graph similarity Alice/Bob:", sim)

    # Arbitrary Cypher: list all entities
    cypher_rows = await graph_run_cypher(db_id, "MATCH (e:Entity) RETURN e")
    print("Entities via Cypher:", cypher_rows)

    # Cleanup when done
    await delete_db(db_id)
    print("Deleted graph/vector namespace:", db_id)


def run_sync() -> None:
    db_id = vg_sync.create_db()
    print(f"Created graph/vector namespace: {db_id}")

    alice_id = str(uuid.uuid4())
    acme_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())

    vg_sync.graph_create_type(db_id, "Person")
    vg_sync.graph_create_type(db_id, "Company")

    vg_sync.graph_create_entity(db_id, alice_id, "Alice", "Engineer at ACME")
    vg_sync.graph_create_entity(db_id, acme_id, "ACME", "A rocket company")
    vg_sync.graph_create_entity(db_id, bob_id, "Bob", "Designer at ACME")

    vg_sync.graph_assign_type(db_id, alice_id, "Person")
    vg_sync.graph_assign_type(db_id, bob_id, "Person")
    vg_sync.graph_assign_type(db_id, acme_id, "Company")

    vg_sync.graph_create_relationship(db_id, alice_id, acme_id, "EMPLOYED_AT", since="2022")
    vg_sync.graph_create_relationship(db_id, bob_id, acme_id, "EMPLOYED_AT", since="2021")
    vg_sync.graph_create_relationship(db_id, alice_id, bob_id, "COLLABORATES_WITH", project="Design")

    vg_sync.vector_add(db_id, alice_id, make_vec(0.1), {"name": "Alice", "team": "eng"})
    vg_sync.vector_add(db_id, bob_id, make_vec(0.11), {"name": "Bob", "team": "design"})
    vg_sync.vector_add(db_id, acme_id, make_vec(0.5), {"name": "ACME", "team": "org"})

    neighbors = vg_sync.vector_nearest_neighbors(db_id, make_vec(0.1), k=2)
    print("Nearest to Alice vector:", neighbors)

    design_only = vg_sync.vector_nearest_neighbors(db_id, make_vec(0.11), k=2, metadata_filter={"team": "design"})
    print("Design-only neighbors:", design_only)

    bob_neighbors = vg_sync.vector_query_by_id(db_id, bob_id, k=2)
    print("Neighbors of Bob (by id):", bob_neighbors)

    collaborators = vg_sync.graph_neighbors(db_id, alice_id, depth=1, relation="COLLABORATES_WITH")
    print("Alice collaborators:", collaborators)

    sim = vg_sync.graph_similarity(db_id, alice_id, bob_id, depth=2)
    print("Graph similarity Alice/Bob:", sim)

    vg_sync.delete_db(db_id)
    print("Deleted graph/vector namespace:", db_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorGraph demo (async by default, sync with --sync)")
    parser.add_argument("--sync", action="store_true", help="Run the synchronous demo")
    args = parser.parse_args()
    if args.sync:
        run_sync()
    else:
        asyncio.run(run_async())
