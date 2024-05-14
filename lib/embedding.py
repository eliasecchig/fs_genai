from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from typing import List
from langchain_core.embeddings import Embeddings



class VertexEmbeddingFunction(Embeddings):
    """Custom Langchain embedding model leveraging GCP embedding"""

    def __init__(self, model_name: str = "textembedding-gecko@003"):
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_query(self, query: str, task="RETRIEVAL_QUERY") -> List[float]:
        """Return the embedding for a query"""
        input = TextEmbeddingInput(query, task)
        embedding = self.model.get_embeddings([input])[0]
        return embedding.values

    def embed_documents(
            self,
            texts: List[str],
            task: str = "RETRIEVAL_DOCUMENT",
            max_batch_size: int = 5
    ) -> List[List[float]]:
        """Return the embeddings for a list of documents"""
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        all_embeddings = []
        for start in range(0, len(inputs), max_batch_size):
            end = start + max_batch_size
            batch = inputs[start: end]
            all_embeddings.extend(self.model.get_embeddings(batch))
        return [embedding.values for embedding in all_embeddings]
