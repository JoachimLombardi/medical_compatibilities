import html
import json
import os
from pathlib import Path
import unicodedata
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()


def normalize_text(text: str) -> str:
    # 1. Decode HTML entities (&#39; → ')
    text = html.unescape(text)
    # 2. Remove page numbers stuck to words (e.g. 14faiblesse)
    text = re.sub(r'\b\d+(?=[a-zA-Z])', '', text)
    # 3. Fix broken line breaks (PDF artifacts)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 4. Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # 5. Preserve paragraph breaks
    text = re.sub(r'\n{2,}', '\n\n', text)
    # 6. Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # 7. Remove leading and trailing whitespace
    text = text.replace("\xa0", " ")
    return text.strip()


def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


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
    points_path = Path("data/json/points.json")
    if points_path.exists():
        with open(points_path, "r") as f:
            points = json.load(f)
    else:
        notices = []
        points = []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for file in Path("data/text").iterdir():
            if not file.is_file():
                continue
            normalized_text = normalize_text(file.read_text())
            medication_name = file.stem
            notices.append({"medication_name": medication_name, "text": normalized_text})
            tokens = encoding.encode(normalized_text)
            start = 0
            while start < len(tokens):
                end = min(start + 1000, len(tokens))
                chunk_text = encoding.decode(tokens[start:end])
                # remove text after last dot
                last_dot = chunk_text.rfind(".")
                if last_dot != -1:
                    chunk_text = chunk_text[:last_dot+1]
                payload = {
                    "medication_name": medication_name,
                    "text": chunk_text,
                    "contraindication": contains_any(chunk_text, ["contre-indication", "allergie", "autres médicaments", 
                                                                  "grossesse", "allaitement", "substance active", 
                                                                  "intolérance", "hypersensibilité", "interference",
                                                                  "incompatibilité", "interaction", "modiffication du mode d'action",]),
                    "adverse_effects": contains_any(chunk_text, ["effets indésirables", "effets secondaires", "risque",
                                                                 "douleur", "grave"])
                }
                embedding = client.embeddings.create(input=chunk_text, model="text-embedding-3-large").data[0].embedding
                points.append({
                    "id": len(points),
                    "vector": embedding,
                    "payload": payload
                })
                start += 900
        with open("data/json/notices.json", "w") as f:
            json.dump(notices, f)
        with open(points_path, "w") as f:
            json.dump(points, f)
#     qdrant =  QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
#     qdrant.create_collection(collection_name="medical_docs",
#       vectors_config=VectorParams(size=len(points[0]["vector"]), distance=Distance.COSINE),
#    )
#     qdrant.upsert(collection_name="medical_docs", points=points)



