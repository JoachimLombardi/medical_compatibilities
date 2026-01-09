import json
import os
from pathlib import Path
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import re

load_dotenv()


def normalize_text(text: str) -> str:
    # 1. Replace newlines with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 2. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # 3. Restore newlines
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


def index_qdrant():
    """
    Index Qdrant with embeddings from medical texts.
    This function indexes a Qdrant collection with embeddings from medical texts.
    It first checks if a file called "json/points.json" exists. If it does, it loads the
    points from that file. Otherwise, it reads all the text files from "data/text", and
    chunks them into chunks of size 1000 with an overlap of 100. It then uses the
    OpenAI API to compute the embeddings of the chunks, and creates a Qdrant collection
    with the embeddings. Finally, it upserts the points into the collection.
    """
    encoding = tiktoken.encoding_for_model("gpt-4.1")
    texts = []
    json_file = Path("data/json/points.json")
    if json_file.exists():
        with open(json_file, "r") as f:
            points = json.load(f)
    else:
        for data in Path("data/text").iterdir():
            if data.is_file():
                texts.append(data.read_text())
        chunks = []
        chunk_size = 1000     
        overlap_size = 100     
        for text in texts:
            text = normalize_text(text)
            tokens = encoding.encode(text)
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                start += chunk_size - overlap_size
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = [
            client.embeddings.create(input=chunk, model="text-embedding-3-large").data[0].embedding
            for chunk in chunks
        ]
        points = [{"id": i, "vector": v, "payload": {"text": chunks[i]}} for i, v in enumerate(embeddings)]
        with open(json_file, "w") as f:
            json.dump(points, f)
    qdrant =  QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    qdrant.create_collection(collection_name="medical_docs",
      vectors_config=VectorParams(size=len(points[0]["vector"]), distance=Distance.COSINE),
   )
    qdrant.upsert(collection_name="medical_docs", points=points)



