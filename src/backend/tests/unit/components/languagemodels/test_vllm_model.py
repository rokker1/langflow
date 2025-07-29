from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI
from langflow.components.vllm.vllm_model import VLLMModelComponent

from tests.base import ComponentTestBaseWithoutClient


class TestVLLMModelComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return VLLMModelComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "base_url": "http://localhost:8000/v1",
            "model_name": "test-model",
            "api_key": "test-key",
            "temperature": 0.1,
            "max_tokens": 100,
            "model_kwargs": {},
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []

    @patch("langflow.components.vllm.vllm_model.ChatOpenAI")
    async def test_build_model(self, mock_chat_openai, component_class, default_kwargs):
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        component = component_class(**default_kwargs)
        model = component.build_model()

        mock_chat_openai.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            max_tokens=100,
            model_kwargs={},
            temperature=0.1,
        )
        assert model == mock_instance

    def test_build_model_integration(self):
        component = VLLMModelComponent()
        component.base_url = "http://localhost:8000/v1"
        component.model_name = "test-model"
        component.temperature = 0.2
        component.max_tokens = 10

        model = component.build_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model == "test-model"
        assert model.base_url == "http://localhost:8000/v1"
