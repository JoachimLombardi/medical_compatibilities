import os
from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import QueryResponse

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))


def search(query: str, k: int=1) -> QueryResponse:
    """
    Search the medical documents collection using the given query.
    The query is embedded into a vector using the OpenAI text-embedding-3-large model.
    The vector is then used to query the medical_docs collection in Qdrant.
    The limit parameter determines the number of results to return.
    Returns a QueryResponse object with the results.
    Args:
        query (str): The query to search for.
        k (int, optional): The number of results to return. Defaults to 1.
    Returns:
        QueryResponse: The results of the query.
    """
    query_vector = client.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
    return qdrant.query_points(collection_name="medical_docs", query=query_vector, limit=k, with_payload=True)