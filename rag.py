import os
from search import search
from index_qdrant import index_qdrant
from generation import generation
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


def rag(query: str) -> tuple[str, str, str]:
    """
    This function takes a query as input and returns a tuple containing the generated response and the retrieved documents.
    It first checks if the collection "medical_docs" exists in Qdrant. If not, it indexes the collection using the index_qdrant function.
    Then it performs retrieval on the query using the search function and generates a response based on the retrieved documents using the generation function.
    Finally, it returns a tuple containing the generated response and the retrieved documents.
    Args:
        query (str): The query to be used for retrieval and generation.
    Returns:
        tuple[str, str, str]: A tuple containing the generated response and the retrieved documents.
    """
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    if not qdrant.collection_exists("medical_docs"):
        index_qdrant()
    retrievals = search(query)
    adverse_effects = "\n".join(retrieval.payload["id"] + "\n" + retrieval.payload["text"] for retrieval in retrievals if retrieval.payload["adverse_effects"])
    contraindication = "\n".join(retrieval.payload["id"] + "\n" + retrieval.payload["text"] for retrieval in retrievals if retrieval.payload["contraindication"])
    response, adverse_effects, evidence_compatibility = generation(query, adverse_effects, contraindication)
    return response, adverse_effects, evidence_compatibility
