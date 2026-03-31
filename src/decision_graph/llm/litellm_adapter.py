import json
import logging
from typing import Any, Optional

from decision_graph.core.config import LLMConfig
from decision_graph.core.interfaces import LLMAdapter

_logger = logging.getLogger(__name__)


class LiteLLMAdapter(LLMAdapter):
    """LLM adapter backed by LiteLLM. Supports 100+ providers out of the box."""

    async def execute_async(self, model_config: LLMConfig, data: Any, additional_data: Optional[dict] = None) -> Any:
        import litellm

        messages = self._build_messages(model_config, data, additional_data)

        kwargs = {
            "model": model_config.model_name,
            "messages": messages,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }
        if model_config.max_tokens:
            kwargs["max_tokens"] = model_config.max_tokens
        if model_config.data_model:
            kwargs["response_format"] = {"type": "json_object"}

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content

        if model_config.data_model:
            return self._parse_structured(content, model_config.data_model)

        return content

    def _build_messages(self, config: LLMConfig, data: Any, additional_data: Optional[dict] = None) -> list:
        messages = []

        system_parts = []
        if config.system_prompt:
            system_parts.append(config.system_prompt)
        if config.data_model:
            schema = config.data_model.model_json_schema()
            system_parts.append(f"Respond with valid JSON matching this schema:\n{json.dumps(schema)}")
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        template_vars = {"text": str(data)}
        if additional_data:
            template_vars.update(additional_data)

        user_content = config.prompt.format(**template_vars) if config.prompt else str(data)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_structured(self, content: str, data_model):
        if hasattr(content, "model_dump"):
            return content
        parsed = json.loads(content)
        return data_model.model_validate(parsed)
