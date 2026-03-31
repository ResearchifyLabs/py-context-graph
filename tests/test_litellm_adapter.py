"""Tests for LiteLLMAdapter — prompt building and structured output parsing."""

import json
import unittest
from unittest.mock import AsyncMock, patch

import pydantic

from decision_graph.core.config import LLMConfig
from decision_graph.llm.litellm_adapter import LiteLLMAdapter


class SampleOutput(pydantic.BaseModel):
    summary: str
    score: int


class TestLiteLLMAdapterMessageBuilding(unittest.TestCase):
    def setUp(self):
        self.adapter = LiteLLMAdapter()

    def test_builds_user_message_from_prompt_template(self):
        config = LLMConfig(prompt="Analyze: {text}")
        messages = self.adapter._build_messages(config, data="hello world")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Analyze: hello world")

    def test_data_model_injects_schema_in_system_message(self):
        config = LLMConfig(prompt="{text}", data_model=SampleOutput)
        messages = self.adapter._build_messages(config, data="test")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("JSON", messages[0]["content"])
        self.assertIn("summary", messages[0]["content"])

    def test_includes_system_prompt(self):
        config = LLMConfig(system_prompt="You are a helpful assistant.", prompt="{text}")
        messages = self.adapter._build_messages(config, data="test")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(messages[1]["role"], "user")

    def test_additional_data_substituted_in_prompt(self):
        config = LLMConfig(prompt="Text: {text}\nContext: {decision_json}")
        messages = self.adapter._build_messages(
            config, data="summary", additional_data={"decision_json": '{"type":"fix"}'}
        )
        self.assertIn('{"type":"fix"}', messages[0]["content"])

    def test_no_prompt_uses_data_directly(self):
        config = LLMConfig()
        messages = self.adapter._build_messages(config, data="raw text")
        self.assertEqual(messages[0]["content"], "raw text")


class TestLiteLLMAdapterStructuredParsing(unittest.TestCase):
    def setUp(self):
        self.adapter = LiteLLMAdapter()

    def test_parses_json_string_to_pydantic(self):
        raw = json.dumps({"summary": "test", "score": 5})
        result = self.adapter._parse_structured(raw, SampleOutput)
        self.assertIsInstance(result, SampleOutput)
        self.assertEqual(result.summary, "test")
        self.assertEqual(result.score, 5)

    def test_returns_pydantic_object_as_is(self):
        obj = SampleOutput(summary="already parsed", score=10)
        result = self.adapter._parse_structured(obj, SampleOutput)
        self.assertIs(result, obj)

    def test_raises_on_invalid_json(self):
        with self.assertRaises(Exception):
            self.adapter._parse_structured("not json", SampleOutput)


class TestLiteLLMAdapterExecution(unittest.IsolatedAsyncioTestCase):
    async def test_calls_litellm_acompletion(self):
        adapter = LiteLLMAdapter()
        config = LLMConfig(
            model_name="gpt-4.1-mini",
            prompt="Summarize: {text}",
            temperature=0.2,
            data_model=SampleOutput,
        )

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = json.dumps({"summary": "done", "score": 9})

        with patch("litellm.acompletion", return_value=mock_response) as mock_acompletion:
            result = await adapter.execute_async(config, data="some text")

            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args[1]
            self.assertEqual(call_kwargs["model"], "gpt-4.1-mini")
            self.assertEqual(call_kwargs["temperature"], 0.2)
            self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
            messages = call_kwargs["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("JSON", messages[0]["content"])
            self.assertIn("Summarize: some text", messages[1]["content"])

        self.assertIsInstance(result, SampleOutput)
        self.assertEqual(result.summary, "done")

    async def test_returns_raw_string_when_no_data_model(self):
        adapter = LiteLLMAdapter()
        config = LLMConfig(prompt="{text}")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "plain response"

        with patch("litellm.acompletion", return_value=mock_response):
            result = await adapter.execute_async(config, data="input")

        self.assertEqual(result, "plain response")
