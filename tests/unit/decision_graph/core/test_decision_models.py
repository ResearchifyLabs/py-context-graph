import unittest

from decision_graph.core.decision_trace_profiles import allowed_decision_types_for_industry
from decision_graph.core.domain import (
    DecisionEnrichmentRow,
    DecisionJoinRow,
    DecisionLineageRow,
    DecisionUnitCoreExtractList,
    DecisionUnitRow,
    DecisionUnitRowList,
)
from decision_graph.core.matching import merge_decision_trace_history


class TestDecisionModels(unittest.TestCase):
    def test_allowed_decision_types_for_industry(self):
        self.assertIn("BugReport", allowed_decision_types_for_industry("product_support"))
        self.assertIn("ClientPreference", allowed_decision_types_for_industry("concierge"))
        self.assertEqual(
            allowed_decision_types_for_industry("unknown_profile"), allowed_decision_types_for_industry(None)
        )

    def test_decision_trace_unit_parses(self):
        unit = DecisionUnitRow.model_validate(
            {
                "decision_id": "abcd1234",
                "pid": "p1",
                "gid": "g1",
                "cid": "c1",
                "updated_at": 1768220580,
                "recorded_at": 1768220581,
                "decision_type": "purchase",
                "initiator_name": "Alex",
                "initiator_role": "internal",
                "counterparty_names": ["Vendor Inc"],
                "subject_label": "Withings Sleep Analyzer",
                "action_type": "request",
                "action_desc": "Request payment approval for 10 units",
                "status_state": "proposed",
                "status_blocker": None,
                "evidence_span": "Please approve payment for 10 units.",
                "confidence": 0.8,
            }
        )
        self.assertEqual(unit.decision_id, "abcd1234")
        self.assertEqual(unit.action_type, "request")

    def test_decision_trace_parses_list(self):
        trace = DecisionUnitRowList.model_validate(
            {
                "items": [
                    {
                        "decision_id": "abcd1234",
                        "pid": "p1",
                        "gid": "g1",
                        "cid": "c1",
                        "updated_at": 1768220580,
                        "recorded_at": 1768220581,
                        "decision_type": "purchase",
                        "initiator_name": "Alex",
                        "initiator_role": "internal",
                        "counterparty_names": ["Vendor Inc"],
                        "subject_label": "Withings Sleep Analyzer",
                        "action_type": "request",
                        "action_desc": "Request payment approval for 10 units",
                        "status_state": "proposed",
                        "status_blocker": None,
                        "evidence_span": "Please approve payment for 10 units.",
                        "confidence": 0.8,
                    }
                ]
            }
        )
        self.assertEqual(len(trace.items), 1)
        self.assertEqual(trace.items[0].decision_id, "abcd1234")

    def test_decision_trace_prompt_a_model_parses(self):
        extracted = DecisionUnitCoreExtractList.model_validate(
            {
                "items": [
                    {
                        "decision_type": "ClientPreference",
                        "actors": {
                            "initiator_name": "Animesh",
                            "initiator_role": "unknown",
                            "counterparty_names": ["family", "Savaan"],
                        },
                        "subject": {
                            "label": "Family vegetarian for 1 month starting 2025-11-17 14:45",
                        },
                        "action": {
                            "type": "announce_change",
                            "description": "Announce family will follow a vegetarian diet for one month due to Savaan.",
                        },
                        "status": {"state": "done", "blocker": None},
                        "evidence": {
                            "span": "Animesh said because of Savaan, family is vegetarian for next 1 month.",
                            "confidence": 0.8,
                        },
                    }
                ]
            }
        )
        self.assertEqual(len(extracted.items), 1)
        self.assertEqual(extracted.items[0].action.type, "announce_change")

    def test_decision_id_signature_is_stable(self):
        from decision_graph.decision_enrichment import DecisionEnrichmentService

        cid = "c1"
        decision_type = "Purchase"
        subject_label = " Withings   Sleep Analyzer "
        initiator_name = "Alex"

        did1 = DecisionEnrichmentService.compute_decision_id(
            cid=cid,
            decision_type=decision_type,
            action_type="request",
            subject_label=subject_label,
            initiator_name=initiator_name,
        )
        did2 = DecisionEnrichmentService.compute_decision_id(
            cid=cid,
            decision_type=decision_type,
            action_type="request",
            subject_label=subject_label,
            initiator_name=initiator_name,
        )
        self.assertEqual(did1, did2)

    def test_decision_id_generation_multiple_cases(self):
        from decision_graph.decision_enrichment import DecisionEnrichmentService

        base = dict(
            cid="c_base",
            decision_type="BugReport",
            action_type="status_update",
            subject_label="TestClientAnalytics.test_action_type_analysis_empty_action_types failure",
            initiator_name="Shivansh Vishwakarma",
        )

        did_base = DecisionEnrichmentService.compute_decision_id(**base)

        # Normalization: case/whitespace differences shouldn't change id.
        did_norm = DecisionEnrichmentService.compute_decision_id(
            cid="c_base",
            decision_type=" bugreport ",
            action_type=" status_update ",
            subject_label="  TestClientAnalytics.test_action_type_analysis_empty_action_types   failure  ",
            initiator_name="  SHIVANSH   VISHWAKARMA ",
        )
        self.assertEqual(did_base, did_norm)

        # Different subject_label => different id.
        did_subject2 = DecisionEnrichmentService.compute_decision_id(
            cid="c_base",
            decision_type="BugReport",
            action_type="status_update",
            subject_label="some other failure",
            initiator_name="Shivansh Vishwakarma",
        )
        self.assertNotEqual(did_base, did_subject2)

        # Different initiator => different id.
        did_initiator2 = DecisionEnrichmentService.compute_decision_id(
            cid="c_base",
            decision_type="BugReport",
            action_type="status_update",
            subject_label=base["subject_label"],
            initiator_name="Someone Else",
        )
        self.assertNotEqual(did_base, did_initiator2)

        # Different decision_type => different id.
        did_type2 = DecisionEnrichmentService.compute_decision_id(
            cid="c_base",
            decision_type="Purchase",
            action_type="status_update",
            subject_label=base["subject_label"],
            initiator_name=base["initiator_name"],
        )
        self.assertNotEqual(did_base, did_type2)

        # Different cid => different id.
        did_cid2 = DecisionEnrichmentService.compute_decision_id(
            cid="c_other",
            decision_type=base["decision_type"],
            action_type=base["action_type"],
            subject_label=base["subject_label"],
            initiator_name=base["initiator_name"],
        )
        self.assertNotEqual(did_base, did_cid2)

        # action_type change => different id (since action_type is part of signature again).
        did_action2 = DecisionEnrichmentService.compute_decision_id(
            cid="c_base",
            decision_type=base["decision_type"],
            action_type="fix",
            subject_label=base["subject_label"],
            initiator_name=base["initiator_name"],
        )
        self.assertNotEqual(did_base, did_action2)

    def test_decision_id_behaves_for_sample_bugreport_rows(self):
        """
        Test based on exported data in decision_enrichments_slim.json (first few rows).

        We intentionally do NOT assert exact decision_id hashes here, because historical stored decision_ids
        may have been computed with an older signature scheme. Instead, we assert the key invariants:
        - Deterministic: same inputs => same id
        - Sensitive to key fields: different action_type/subject_label => different id
        """
        from decision_graph.decision_enrichment import DecisionEnrichmentService

        samples = [
            {
                "cid": "561d1e59b8b64a4fbf5aaaaf54efbc57",
                "decision_type": "BugReport",
                "action_type": "fix",
                "subject_label": "test_action_type_analysis_empty_action_types failure in Client Analytics",
                "initiator_name": "Shivansh Vishwakarma",
            },
            {
                "cid": "561d1e59b8b64a4fbf5aaaaf54efbc57",
                "decision_type": "BugReport",
                "action_type": "status_update",
                "subject_label": "test_action_type_analysis_empty_action_types failure in Client Analytics",
                "initiator_name": "Shivansh Vishwakarma",
            },
            {
                "cid": "561d1e59b8b64a4fbf5aaaaf54efbc57",
                "decision_type": "BugReport",
                "action_type": "status_update",
                "subject_label": "test_action_type_analysis_empty_action_types failure in Client Analytics module",
                "initiator_name": "Shivansh Vishwakarma",
            },
            {
                "cid": "561d1e59b8b64a4fbf5aaaaf54efbc57",
                "decision_type": "BugReport",
                "action_type": "status_update",
                "subject_label": "Client Analytics test failure on action_type_analysis_empty_action_types",
                "initiator_name": "Shivansh Vishwakarma",
            },
            {
                "cid": "561d1e59b8b64a4fbf5aaaaf54efbc57",
                "decision_type": "BugReport",
                "action_type": "status_update",
                "subject_label": "TestClientAnalytics.test_action_type_analysis_empty_action_types failure",
                "initiator_name": "Shivansh Vishwakarma",
            },
        ]

        computed = []
        for s in samples:
            did1 = DecisionEnrichmentService.compute_decision_id(
                cid=s["cid"],
                decision_type=s["decision_type"],
                action_type=s["action_type"],
                subject_label=s["subject_label"],
                initiator_name=s["initiator_name"],
            )
            did2 = DecisionEnrichmentService.compute_decision_id(
                cid=s["cid"],
                decision_type=s["decision_type"],
                action_type=s["action_type"],
                subject_label=s["subject_label"],
                initiator_name=s["initiator_name"],
            )
            self.assertEqual(did1, did2)
            computed.append(did1)

        # Same cid/decision_type/subject/initiator but different action_type => different id.
        self.assertNotEqual(computed[0], computed[1])

        # But different subject_label should still yield different ids.
        self.assertNotEqual(computed[1], computed[2])

    def test_decision_trace_history_appends_and_dedupes(self):
        base_proj = {"proj_id": "d1"}

        e1 = {
            "updated_at": 1.0,
            "recorded_at": 100,
            "action_type": "status_update",
            "status_state": "proposed",
            "status_blocker": None,
        }
        out1 = merge_decision_trace_history(existing_proj={}, projection=dict(base_proj), event=e1)
        self.assertEqual(out1["status_history"], [e1])
        self.assertEqual(out1["action_type_history"], ["status_update"])

        out2 = merge_decision_trace_history(existing_proj=out1, projection=dict(base_proj), event=e1)
        self.assertEqual(out2["status_history"], [e1])
        self.assertEqual(out2["action_type_history"], ["status_update"])

        e2 = {**e1, "updated_at": 2.0, "status_state": "done"}
        out3 = merge_decision_trace_history(existing_proj=out2, projection=dict(base_proj), event=e2)
        self.assertEqual(out3["status_history"], [e1, e2])
        self.assertEqual(out3["action_type_history"], ["status_update"])

        e3 = {**e2, "updated_at": 3.0, "action_type": "fix"}
        out4 = merge_decision_trace_history(existing_proj=out3, projection=dict(base_proj), event=e3)
        self.assertEqual(out4["status_history"], [e1, e2, e3])
        self.assertEqual(out4["action_type_history"], ["status_update", "fix"])

    def test_decision_enrichment_row_parses(self):
        enr = DecisionEnrichmentRow.model_validate(
            {
                "decision_id": "abcd1234",
                "recorded_at": 1768220581,
                "topics": ["procurement"],
                "entities": [{"type": "Product", "name": "Withings Sleep Analyzer"}],
                "action_params": {"price_inr": 25000, "quantity": 10},
                "constraints_text": ["Minimum order 6 units"],
                "key_facts": [{"k": "price", "v": "25000 INR"}],
                "constraint_evaluations": [],
                "alternatives_considered": [],
                "selected_alternative": None,
                "rejection_reasons": [],
            }
        )
        self.assertEqual(enr.entities[0].type, "Product")
        self.assertEqual(enr.action_params["quantity"], 10)

    def test_flattened_inputs_coerce(self):
        unit = DecisionUnitRow.model_validate(
            {
                "decision_id": "d1",
                "pid": "p1",
                "gid": "g1",
                "cid": "c1",
                "updated_at": 1768220580,
                "recorded_at": 1768220581,
                "decision_type": "ClientPreference",
                "status_state": "done",
                "status_blocker": None,
                "initiator_name": "Animesh",
                "initiator_role": "unknown",
                "counterparty_names": "family, Savaan",
                "subject_label": "Family vegetarian for 1 month starting 2025-11-17 14:45",
                "action_type": "announce_change",
                "action_desc": "Announce family will follow a vegetarian diet for one month due to Savaan.",
                "evidence_span": "Animesh Said the because of Savaan, family is vegetarian for next 1 month.",
                "confidence": 0.8,
            }
        )
        self.assertEqual(unit.status_state, "done")
        self.assertEqual(unit.counterparty_names, ["family", "Savaan"])

        enr = DecisionEnrichmentRow.model_validate(
            {
                "decision_id": "d1",
                "recorded_at": 1768220581,
                "topics": "vegetarian; family; Savaan; 1 month",
                "entities": "Person:Animesh; Other:Savaan; Other:family",
                "action_params": "start=2025-11-17 14:45; duration=1 month; reason=Savaan; affected=family",
                "constraints_text": "starts 2025-11-17 14:45; duration 1 month",
                "key_facts": "reason=Savaan; start=2025-11-17 14:45; duration=1 month; affected=family",
            }
        )
        self.assertEqual(enr.topics[:2], ["vegetarian", "family"])
        self.assertEqual(enr.entities[0].name, "Animesh")
        self.assertEqual(enr.action_params["duration"], "1 month")
        self.assertEqual(enr.constraints_text[0], "starts 2025-11-17 14:45")
        self.assertEqual(enr.key_facts[0].k, "reason")

    def test_decision_join_row_parses(self):
        row = DecisionJoinRow.model_validate(
            {
                "from_decision_id": "a",
                "to_decision_id": "b",
                "join_type": "same_intent",
                "join_confidence": 0.9,
                "join_reasons": ["subject overlap"],
                "created_at": 1768220581,
            }
        )
        self.assertEqual(row.join_type, "same_intent")

    def test_decision_lineage_row_parses(self):
        row = DecisionLineageRow.model_validate(
            {
                "from_decision_id": "b",
                "to_decision_id": "a",
                "edge_type": "supersedes",
                "confidence": 0.8,
                "rationale": "Later decision marked done.",
                "created_at": 1768220581,
            }
        )
        self.assertEqual(row.edge_type, "supersedes")
