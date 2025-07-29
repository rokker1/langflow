from typing import Any

from langchain_openai import ChatOpenAI
from typing_extensions import override

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs.inputs import DictInput, FloatInput, IntInput, SecretStrInput, StrInput


class VLLMModelComponent(LCModelComponent):
    """Component to interact with a vLLM server using the OpenAI API."""

    display_name = "vLLM"
    description = "Generate text using a vLLM server."
    icon = "VLLM"
    name = "VLLMModel"

    @override
    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        return build_config

    inputs = [
        *LCModelComponent._base_inputs,
        StrInput(
            name="base_url",
            display_name="Base URL",
            advanced=False,
            info="Endpoint of the vLLM API. Defaults to 'http://localhost:8000/v1'.",
            value="http://localhost:8000/v1",
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            info="Name of the model to use.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key for the vLLM server if required.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="Maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            advanced=True,
        ),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        return ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url or "http://localhost:8000/v1",
            api_key=self.api_key,
            max_tokens=self.max_tokens or None,
            model_kwargs=self.model_kwargs or {},
            temperature=self.temperature if self.temperature is not None else 0.7,
        )
