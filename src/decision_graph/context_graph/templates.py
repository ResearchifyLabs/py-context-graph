"""
Parameterized Cypher query templates for context graph queries.
"""

RESOLVE_NODES = """
WITH toLower($q) AS q
MATCH (n)
WHERE any(l IN labels(n) WHERE l IN $types)
  AND toLower(coalesce(n.name, "")) CONTAINS q
RETURN n, labels(n) AS lbls
LIMIT $top_k
"""

RESOLVE_BY_KEYWORD = """
WITH toLower($q) AS q
MATCH (d:Decision)-[:HAS_FACT]->(f:Fact)
WHERE toLower(coalesce(f.v, "")) CONTAINS q
   OR toLower(coalesce(f.k, "")) CONTAINS q
RETURN DISTINCT d, labels(d) AS lbls
ORDER BY d.linked_at DESC
LIMIT $top_k
"""

TOPIC_RELATED_DECISIONS_V1 = """
WITH toLower($q) AS q
MATCH (t:Topic)
WHERE toLower(coalesce(t.name, "")) CONTAINS q
   OR q CONTAINS toLower(coalesce(t.name, ""))

MATCH pathD = (t)<-[:HAS_TOPIC]-(d:Decision)
OPTIONAL MATCH pathP = (d)-[:INITIATED_BY]->(p:Entity)
OPTIONAL MATCH pathE = (d)-[:MENTIONS_ENTITY]->(e:Entity)
OPTIONAL MATCH pathC = (d)-[:HAS_COUNTERPARTY]->(cp:Entity)

WITH t, d, pathD, pathP, pathE, pathC
ORDER BY d.linked_at DESC
LIMIT $max_paths

RETURN t AS seed,
       collect(DISTINCT pathD) AS primary_paths,
       collect(DISTINCT pathP) AS initiator_paths,
       collect(DISTINCT pathE) AS entity_paths,
       collect(DISTINCT pathC) AS counterparty_paths
"""

PERSON_RELATED_DECISIONS_V1 = """
WITH toLower($q) AS q
MATCH (p:Entity)
WHERE p.type = 'Person'
  AND (toLower(coalesce(p.name, "")) CONTAINS q
       OR q CONTAINS toLower(coalesce(p.name, "")))

MATCH pathD = (p)<-[r]-(d:Decision)
WHERE type(r) IN $rel_allowlist
OPTIONAL MATCH pathT = (d)-[:HAS_TOPIC]->(t:Topic)
OPTIONAL MATCH pathE = (d)-[:MENTIONS_ENTITY]->(e:Entity)
OPTIONAL MATCH pathC = (d)-[:HAS_COUNTERPARTY]->(cp:Entity)

WITH p, d, pathD, pathT, pathE, pathC
ORDER BY d.linked_at DESC
LIMIT $max_paths

RETURN p AS seed,
       collect(DISTINCT pathD) AS primary_paths,
       collect(DISTINCT pathT) AS topic_paths,
       collect(DISTINCT pathE) AS entity_paths,
       collect(DISTINCT pathC) AS counterparty_paths
"""

ENTITY_RELATED_DECISIONS_V1 = """
WITH toLower($q) AS q
MATCH (e:Entity)
WHERE toLower(coalesce(e.name, "")) CONTAINS q
   OR q CONTAINS toLower(coalesce(e.name, ""))

MATCH pathD = (e)<-[r]-(d:Decision)
WHERE type(r) IN $rel_allowlist
OPTIONAL MATCH pathT = (d)-[:HAS_TOPIC]->(t:Topic)
OPTIONAL MATCH pathP = (d)-[:INITIATED_BY]->(p:Entity)
OPTIONAL MATCH pathE2 = (d)-[:MENTIONS_ENTITY]->(e2:Entity)
OPTIONAL MATCH pathC = (d)-[:HAS_COUNTERPARTY]->(cp:Entity)

WITH e, d, pathD, pathT, pathP, pathE2, pathC
ORDER BY d.linked_at DESC
LIMIT $max_paths

RETURN e AS seed,
       collect(DISTINCT pathD) AS primary_paths,
       collect(DISTINCT pathT) AS topic_paths,
       collect(DISTINCT pathP) AS initiator_paths,
       collect(DISTINCT pathE2) AS entity_paths,
       collect(DISTINCT pathC) AS counterparty_paths
"""

DECISION_EGO_V1 = """
MATCH (d:Decision {decision_id: $node_id})
OPTIONAL MATCH path = (d)-[r]->(n)
WHERE type(r) IN $rel_allowlist
RETURN d AS seed,
       collect(DISTINCT path) AS primary_paths
"""

EXPLAIN_CONNECTION_V1 = """
MATCH (a {decision_id: $from_id}), (b {decision_id: $to_id})
MATCH p = shortestPath((a)-[*..5]-(b))
RETURN collect(p) AS paths
LIMIT $max_paths
"""

SEARCH_DECISIONS = """
CALL {
  MATCH (t:Topic)<-[:HAS_TOPIC]-(d:Decision)
  WHERE d.gid IN $gids AND toLower(coalesce(t.name, '')) CONTAINS toLower($q)
  RETURN d.decision_id AS decision_id, d.linked_at AS linked_at
  UNION
  MATCH (e:Entity)<-[:MENTIONS_ENTITY|INITIATED_BY]-(d:Decision)
  WHERE d.gid IN $gids AND toLower(coalesce(e.name, '')) CONTAINS toLower($q)
  RETURN d.decision_id AS decision_id, d.linked_at AS linked_at
  UNION
  MATCH (d:Decision)-[:HAS_FACT]->(f:Fact)
  WHERE d.gid IN $gids
    AND (toLower(coalesce(f.v, '')) CONTAINS toLower($q)
         OR toLower(coalesce(f.k, '')) CONTAINS toLower($q))
  RETURN d.decision_id AS decision_id, d.linked_at AS linked_at
}
WITH decision_id, max(linked_at) AS latest
ORDER BY latest DESC
LIMIT $limit
RETURN decision_id
"""

TEMPLATES = {
    "TOPIC_RELATED_DECISIONS_V1": TOPIC_RELATED_DECISIONS_V1,
    "PERSON_RELATED_DECISIONS_V1": PERSON_RELATED_DECISIONS_V1,
    "ENTITY_RELATED_DECISIONS_V1": ENTITY_RELATED_DECISIONS_V1,
    "DECISION_EGO_V1": DECISION_EGO_V1,
    "EXPLAIN_CONNECTION_V1": EXPLAIN_CONNECTION_V1,
}
