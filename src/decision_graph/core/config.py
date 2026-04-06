from dataclasses import dataclass
from typing import Optional, Type

import pydantic

from decision_graph.core.domain import ClusterMetadataExtract


@dataclass
class LLMConfig:
    """Declarative config for a single LLM call. Adapter implementations translate this
    into their provider-specific format (e.g. OpenAI, LiteLLM)."""

    model_name: str = "gpt-4.1-mini"
    prompt: str = ""
    system_prompt: str = ""
    temperature: float = 0.0
    data_model: Optional[Type[pydantic.BaseModel]] = None
    max_tokens: int = 0
    top_p: float = 0.8


cluster_metadata_prompt = """You are analyzing a group of related decisions from business conversations.

Given the following decision information, generate:
1. primary_subject: A concise title (max 10 words) that captures the main topic/thread
2. rolling_summary: A one-line summary (max 30 words) describing the current state and key decisions

Decision Information:
{text}

Focus on:
- The core subject matter being discussed
- Key stakeholders involved
- Current status or outcome
- Any important constraints or parameters"""


cluster_metadata_cfg = LLMConfig(
    model_name="gpt-4.1-mini",
    prompt=cluster_metadata_prompt,
    data_model=ClusterMetadataExtract,
    temperature=0.2,
)
