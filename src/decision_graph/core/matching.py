import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Set

TOPIC_WEIGHT = 0.30
NON_PERSON_ENTITY_WEIGHT = 0.15
PERSON_ENTITY_WEIGHT = 0.05
SUBJECT_WEIGHT = 0.35
SPECIFICS_WEIGHT = 0.15

SPECIFICS_MISMATCH_PENALTY = 0.85
ENTITY_MISMATCH_PENALTY = 0.85

_SPECIFICS_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
_SPECIFICS_STOP_WORDS = frozenset({
    "an", "as", "at", "be", "by", "do", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "been", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "can", "not", "but", "also", "just",
    "than", "then", "only", "very", "its", "his", "her", "they",
    "them", "their", "our", "your", "into", "about", "after", "before",
    "being", "both", "each", "how", "more", "other", "some", "such",
    "who", "whom", "which", "what", "when", "where", "while",
    "all", "any", "did", "does", "doing", "done", "via", "per",
    "confirmed", "shared", "requested", "noted", "announced",
    "approved", "submitted", "reported", "mentioned", "discussed",
})


def dedupe_list(items: Optional[list]) -> list:
    out = []
    seen = set()
    for it in items or []:
        if isinstance(it, dict):
            key = tuple(sorted((str(k), str(v)) for k, v in it.items()))
        else:
            key = str(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def normalize_component(v: Optional[str]) -> str:
    v = (v or "").strip().lower()
    v = v.replace("|", " ")
    v = re.sub(r"\s+", " ", v)
    return v


def compute_decision_id(
    *,
    cid: str,
    decision_type: Optional[str],
    action_type: Optional[str],
    subject_label: Optional[str],
    initiator_name: Optional[str],
) -> str:
    signature = "|".join(
        [
            str(cid),
            normalize_component(decision_type),
            normalize_component(action_type),
            normalize_component(subject_label),
            normalize_component(initiator_name),
        ]
    )
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def merge_enrichment(existing: Optional[dict], incoming: dict) -> dict:
    existing = existing or {}
    merged = {**existing, **incoming}

    if incoming.get("trace_id"):
        merged["trace_id"] = incoming["trace_id"]

    for lf in ("topics", "entities", "constraints_text", "key_facts"):
        merged[lf] = dedupe_list((existing.get(lf) or []) + (incoming.get(lf) or []))

    merged["action_params"] = {
        **(existing.get("action_params") or {}),
        **(incoming.get("action_params") or {}),
    }

    ex_ts = existing.get("recorded_at")
    in_ts = incoming.get("recorded_at")
    if ex_ts is None:
        merged["recorded_at"] = in_ts
    elif in_ts is None:
        merged["recorded_at"] = ex_ts
    else:
        merged["recorded_at"] = max(float(ex_ts), float(in_ts))

    return merged


def normalize_entity(entity: Optional[str]) -> str:
    if not entity:
        return ""
    normalized = entity.lower().strip()
    normalized = re.sub(r'\.(com|io|org|net|co|ai)$', '', normalized)
    normalized = normalized.replace('-', '').replace('_', '').replace(' ', '')
    return normalized


def calculate_jaccard(
    set1: List[str],
    set2: List[str],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> float:
    if not set1 and not set2:
        return 0.0
    if normalize_fn:
        s1 = set(normalize_fn(s) for s in set1 if s)
        s2 = set(normalize_fn(s) for s in set2 if s)
    else:
        s1 = set(s.lower().strip() for s in set1 if s)
        s2 = set(s.lower().strip() for s in set2 if s)
    if not s1 and not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def calculate_sequence_match(str1: str, str2: str) -> float:
    if not str1 and not str2:
        return 0.0
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = {t for t in (a or "").split(" ") if t}
    b_tokens = {t for t in (b or "").split(" ") if t}
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def canonicalize_subject_label(subject_label: Optional[str]) -> str:
    s = (subject_label or "").strip().lower()
    if not s:
        return ""

    s = re.sub(r"https?://\S+", " <url> ", s)
    s = re.sub(r"\bpull/\s*(\d+)\b", r" pr\1 ", s)
    s = re.sub(r"\bpr\s*#?\s*(\d+)\b", r" pr\1 ", s)

    s = s.replace("'", " ").replace('"', " ")

    s = re.sub(r"\b(next day|tomorrow|later|immediately|right away|soon)\b", " ", s)
    s = re.sub(r"\b(recurring|recur|reoccurring)\b", " ", s)

    s = re.sub(r"[^a-z0-9_\.<> ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    tokens = [t for t in s.split(" ") if t and t not in stop]

    noise = {"module", "service", "dashboard", "endpoint", "function", "functions"}
    tokens = [t for t in tokens if t not in noise]

    return " ".join(tokens)


def build_conversation_fingerprint(
    decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not decisions:
        return {}

    all_topics = []
    all_entities = []
    all_subject_labels = []
    all_decision_ids = []
    all_trace_ids = []

    gid = decisions[0].get("gid", "")
    cid = decisions[0].get("cid", "")

    for d in decisions:
        all_topics.extend(d.get("topics", []))
        all_entities.extend(d.get("entities", []))
        if d.get("subject_label"):
            all_subject_labels.append(d["subject_label"])
        if d.get("decision_id"):
            all_decision_ids.append(d["decision_id"])
        if d.get("trace_id"):
            all_trace_ids.append(d["trace_id"])

    topics_deduped = list(set(t.lower().strip() for t in all_topics if t))
    entities_deduped = list(set(e for e in all_entities if e))
    subject_labels_deduped = list(set(all_subject_labels))

    return {
        "gid": gid,
        "cid": cid,
        "topics": topics_deduped,
        "entities": entities_deduped,
        "subject_labels": subject_labels_deduped,
        "decision_ids": all_decision_ids,
        "trace_ids": all_trace_ids,
        "decision_count": len(decisions),
    }


def merge_decision_trace_history(*, existing_proj: dict, projection: dict, event: dict) -> dict:
    action_type = event.get("action_type")

    history = existing_proj.get("status_history") or []
    if not history or history[-1] != event:
        history = list(history) + [event]
    projection["status_history"] = history[-50:]

    at_hist = existing_proj.get("action_type_history") or []
    if action_type and action_type not in at_hist:
        at_hist = list(at_hist) + [action_type]
    projection["action_type_history"] = at_hist[-20:]
    return projection


def calculate_max_subject_match(
    subject_labels_a: List[str],
    subject_labels_b: List[str],
) -> float:
    if not subject_labels_a or not subject_labels_b:
        return 0.0

    max_score = 0.0
    for label_a in subject_labels_a:
        for label_b in subject_labels_b:
            score = calculate_sequence_match(label_a, label_b)
            if score > max_score:
                max_score = score
    return max_score


def tokenize_specifics(raw_values: List[str]) -> Set[str]:
    tokens: Set[str] = set()
    for v in raw_values:
        if not isinstance(v, str):
            continue
        for tok in _SPECIFICS_TOKEN_RE.findall(v.lower()):
            if len(tok) >= 2 and tok not in _SPECIFICS_STOP_WORDS:
                tokens.add(tok)
    return tokens


def precompute_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    topics = decision.get("topics") or []
    person_ents = decision.get("person_entities") or []
    non_person_ents = decision.get("non_person_entities") or []
    key_facts = decision.get("key_facts") or []
    action_params = decision.get("action_params") or {}

    specifics_raw = list(key_facts)
    for k, v in action_params.items():
        if isinstance(v, str) and k not in ("action", "action_type", "action_desc"):
            specifics_raw.append(v)

    specifics_tokens = tokenize_specifics(specifics_raw)
    already_scored = (
        tokenize_specifics(topics)
        | tokenize_specifics(person_ents)
        | tokenize_specifics(non_person_ents)
    )
    specifics_tokens -= already_scored

    return {
        **decision,
        "_topics_set": tokenize_specifics(topics),
        "_person_set": set(normalize_entity(e) for e in person_ents if e),
        "_non_person_set": set(normalize_entity(e) for e in non_person_ents if e),
        "_subject_lower": (decision.get("subject_label") or "").lower().strip(),
        "_specifics_set": specifics_tokens,
    }


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


idf_weighted_jaccard = jaccard


def score_decision_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, float]:
    a_topics = a.get("_topics_set") or set()
    b_topics = b.get("_topics_set") or set()
    topic_score = jaccard(a_topics, b_topics)

    a_np = a.get("_non_person_set") or set()
    b_np = b.get("_non_person_set") or set()
    non_person_score = jaccard(a_np, b_np)

    a_p = a.get("_person_set") or set()
    b_p = b.get("_person_set") or set()
    person_score = jaccard(a_p, b_p)

    subject_score = calculate_sequence_match(
        a.get("_subject_lower") or a.get("subject_label", ""),
        b.get("_subject_lower") or b.get("subject_label", ""),
    )

    a_spec = a.get("_specifics_set") or set()
    b_spec = b.get("_specifics_set") or set()
    specifics_score = jaccard(a_spec, b_spec)

    combined = (
        topic_score * TOPIC_WEIGHT
        + non_person_score * NON_PERSON_ENTITY_WEIGHT
        + person_score * PERSON_ENTITY_WEIGHT
        + subject_score * SUBJECT_WEIGHT
        + specifics_score * SPECIFICS_WEIGHT
    )

    specifics_penalty = 1.0
    entity_penalty = 1.0
    person_penalty = 1.0
    if a_spec and b_spec and specifics_score < 0.10:
        specifics_penalty = SPECIFICS_MISMATCH_PENALTY
    if a_np and b_np and non_person_score < 0.10:
        entity_penalty = ENTITY_MISMATCH_PENALTY
    if a_p and b_p and person_score < 0.10:
        person_penalty = ENTITY_MISMATCH_PENALTY
    combined *= specifics_penalty * entity_penalty * person_penalty

    return {
        "subject_score": round(subject_score, 4),
        "topic_score": round(topic_score, 4),
        "non_person_entity_score": round(non_person_score, 4),
        "person_entity_score": round(person_score, 4),
        "entity_score": round(non_person_score * 0.83 + person_score * 0.17, 4),
        "specifics_score": round(specifics_score, 4),
        "specifics_penalty": round(specifics_penalty, 2),
        "entity_penalty": round(entity_penalty, 2),
        "person_penalty": round(person_penalty, 2),
        "combined_score": round(combined, 4),
    }
