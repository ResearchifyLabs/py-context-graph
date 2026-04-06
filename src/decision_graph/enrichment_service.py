import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List

from decision_graph.core.config import LLMConfig
from decision_graph.core.domain import (
    DecisionEnrichmentCoreExtract,
    DecisionEnrichmentRow,
    DecisionUnitRow,
)
from decision_graph.core.interfaces import LLMAdapter
from decision_graph.core.matching import (
    compute_decision_id,
    dedupe_list,
    merge_enrichment,
    normalize_component,
)
from decision_graph.core.registry import StorageBackend
from decision_graph.prompt_loader import load_prompt

_logger = logging.getLogger(__name__)


class DecisionEnrichmentService:
    def __init__(self, *, backend: StorageBackend, executor: LLMAdapter):
        self._enrichment_store = backend.enrichment_store()
        self._executor = executor
        self._prompt_loader = load_prompt

    @staticmethod
    def _dedupe_list(items):
        return dedupe_list(items)

    @staticmethod
    def _normalize_component(v: str | None) -> str:
        return normalize_component(v)

    @classmethod
    def compute_decision_id(
        cls,
        *,
        cid: str,
        decision_type: str | None,
        action_type: str | None,
        subject_label: str | None,
        initiator_name: str | None,
    ) -> str:
        return compute_decision_id(
            cid=cid,
            decision_type=decision_type,
            action_type=action_type,
            subject_label=subject_label,
            initiator_name=initiator_name,
        )

    @classmethod
    def merge_enrichment(cls, existing: dict | None, incoming: dict) -> dict:
        return merge_enrichment(existing=existing, incoming=incoming)

    async def run_enrichment(
        self,
        *,
        summary_text: str,
        decision_rows: List[DecisionUnitRow],
        gid: str,
        cid: str,
    ) -> List[DecisionEnrichmentRow]:
        if not decision_rows:
            return []

        _logger.info("decision_enrichment.run_start gid=%s cid=%s decisions=%s", gid, cid, len(decision_rows))

        decision_ids = [r.decision_id for r in decision_rows]
        existing_docs = self._enrichment_store.find_by_ids(decision_ids)

        out: List[DecisionEnrichmentRow] = []
        for r in decision_rows:
            try:
                row = await self._run_enrichment_for_decision(summary_text=summary_text, decision_row=r)
            except Exception:
                _logger.exception(
                    "decision_enrichment.llm_failed gid=%s cid=%s decision_id=%s",
                    gid,
                    cid,
                    getattr(r, "decision_id", None),
                )
                continue

            decision_id = row.decision_id
            incoming = {
                "decision_id": decision_id,
                "gid": gid,
                "cid": cid,
                "trace_id": row.trace_id,
                **row.model_dump(exclude={"trace_id", "decision_id"}),
            }
            try:
                existing = existing_docs.get(decision_id)
                if not existing:
                    self._enrichment_store.save(decision_id, incoming)
                    _logger.info(
                        "decision_enrichment.created gid=%s cid=%s decision_id=%s trace_id=%s",
                        gid,
                        cid,
                        decision_id,
                        row.trace_id,
                    )
                else:
                    merged = merge_enrichment(existing=existing, incoming=incoming)

                    changed = False
                    for lf in ("topics", "entities", "constraints_text", "key_facts"):
                        if len(existing.get(lf) or []) != len(merged.get(lf) or []):
                            changed = True
                            break
                    if not changed and (existing.get("action_params") or {}) != (merged.get("action_params") or {}):
                        changed = True

                    self._enrichment_store.upsert(decision_id, merged)
                    if changed:
                        _logger.info(
                            "decision_enrichment.merged gid=%s cid=%s decision_id=%s trace_id=%s topics=%s->%s entities=%s->%s",
                            gid,
                            cid,
                            decision_id,
                            row.trace_id,
                            len(existing.get("topics") or []),
                            len(merged.get("topics") or []),
                            len(existing.get("entities") or []),
                            len(merged.get("entities") or []),
                        )
            except Exception:
                _logger.exception(
                    "decision_enrichment.persist_failed gid=%s cid=%s decision_id=%s trace_id=%s",
                    gid,
                    cid,
                    decision_id,
                    row.trace_id,
                )
                continue

            out.append(row)

        _logger.info(
            "decision_enrichment.run_done gid=%s cid=%s enriched=%s attempted=%s",
            gid,
            cid,
            len(out),
            len(decision_rows),
        )
        return out

    async def _run_enrichment_for_decision(
        self,
        *,
        summary_text: str,
        decision_row: DecisionUnitRow,
    ) -> DecisionEnrichmentRow:
        prompt = self._prompt_loader("decision_enrichment")
        model_cfg = LLMConfig(
            model_name="gpt-4.1-mini",
            prompt=prompt,
            temperature=0.0,
            data_model=DecisionEnrichmentCoreExtract,
        )

        decision_json = json.dumps(
            {
                "decision_type": decision_row.decision_type,
                "subject_label": decision_row.subject_label,
                "action_type": decision_row.action_type,
                "action_desc": decision_row.action_desc,
                "initiator_name": decision_row.initiator_name,
            }
        )

        resp = await self._executor.execute_async(
            model_cfg,
            data=summary_text,
            additional_data={"decision_json": decision_json},
        )

        recorded_at = datetime.now(timezone.utc).timestamp()
        trace_id = uuid.uuid4().hex
        if not resp:
            return DecisionEnrichmentRow(
                decision_id=decision_row.decision_id,
                trace_id=trace_id,
                recorded_at=recorded_at,
                topics=[],
                entities=[],
                action_params={},
                constraints_text=[],
                key_facts=[],
            )

        action_params = {kv.k: kv.v for kv in (resp.action_params or [])}
        return DecisionEnrichmentRow(
            decision_id=decision_row.decision_id,
            trace_id=trace_id,
            recorded_at=recorded_at,
            topics=resp.topics,
            entities=resp.entities,
            action_params=action_params,
            constraints_text=resp.constraints_text,
            key_facts=resp.key_facts,
        )
