from unittest.mock import MagicMock, patch

from langflow.components.localai import LocalAIEmbeddingsComponent


def test_localai_embeddings_returns_vectors():
    texts = ["Hello", "World", "Langflow"]

    with patch("langflow.components.localai.localai.OpenAIEmbeddings") as mock_openai_embeddings:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.0] * 1024 for _ in texts]
        mock_openai_embeddings.return_value = mock_instance

        component = LocalAIEmbeddingsComponent(
            model="bge-m3",
            openai_api_base="http://87.242.104.103:8080/v1",
            openai_api_key="key",
            dimensions=1024,
        )

        embeddings = component.build_embeddings()
        result = embeddings.embed_documents(texts)

        args, kwargs = mock_openai_embeddings.call_args
        assert kwargs["model"] == "bge-m3"
        assert kwargs["base_url"] == "http://87.242.104.103:8080/v1"
        assert kwargs["api_key"] == "key"
        assert kwargs["dimensions"] == 1024

        mock_instance.embed_documents.assert_called_once_with(texts)
        assert len(result) == 3
        assert all(len(vec) == 1024 for vec in result)
