import logging
from typing import List, Optional

from decision_graph.core.config import LLMConfig
from decision_graph.core.decision_trace_profiles import allowed_decision_types_for_industry
from decision_graph.core.domain import DecisionUnitCoreExtractList
from decision_graph.core.interfaces import LLMAdapter
from decision_graph.prompt_loader import load_prompt

_logger = logging.getLogger(__name__)


class DecisionExtractionService:
    """Extracts structured decision items from conversation text via LLM."""

    def __init__(self, *, executor: LLMAdapter):
        self._executor = executor
        self._prompt_loader = load_prompt

    async def extract(
        self,
        conv_text: str,
        *,
        allowed_decision_types: Optional[List[str]] = None,
        domain_hint: str = "NA",
        industry: str = "generic_b2b",
    ) -> List[dict]:
        if allowed_decision_types is None:
            allowed_decision_types = allowed_decision_types_for_industry(industry)

        allowed_str = "\n".join(f"- {t}" for t in allowed_decision_types)
        system_prompt = self._prompt_loader(
            "decision_trace",
            ALLOWED_DECISION_TYPES=allowed_str,
            DOMAIN_HINT=domain_hint,
        )
        system_prompt += "\n\n*No prior decision_trace projections available. Please generate a new list.*"

        model_cfg = LLMConfig(
            model_name="gpt-4.1-mini",
            system_prompt=system_prompt,
            prompt="\n*Conversation (Messages):* \n{text}",
            temperature=0,
            data_model=DecisionUnitCoreExtractList,
        )

        response = await self._executor.execute_async(model_cfg, data=conv_text)
        items = response.model_dump(by_alias=True).get("items", [])
        _logger.info("decision_extraction.done items=%d", len(items))
        return items
