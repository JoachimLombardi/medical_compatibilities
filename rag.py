import json
from generation import generation
from dotenv import load_dotenv

load_dotenv()


def rag(query: str) -> tuple[str, list[str]]:
    """
    This function takes a query as input and returns a tuple containing the response, evidence for compatibility.
    It first checks if the "medical_docs" collection exists in Qdrant. If it doesn't, it calls the index_qdrant function to index the collection.
    It then reads the points and notices from the JSON files and iterates over the notices. 
    For each notice, it constructs a contraindication string by concatenating the text of the points with the same medication name.
    It then calls the generation function with the query and contraindication string, 
    and appends the response and evidence for compatibility to the final response and evidence list.
    Finally, it returns the final response and evidence for compatibility.
    Parameters:
        query (str): The query to search for.
    Returns:
        tuple[str, list[str]]: A tuple containing the response and evidence for compatibility.
    """
    # qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    # qdrant.delete_collection("medical_docs")
    # if not qdrant.collection_exists("medical_docs"):
    #     index_qdrant()
    # retrievals = search(query)
    with open("data/json/points.json", "r") as f:
        points = json.load(f)
    with open("data/json/notices.json", "r") as f:
        notices = json.load(f)
    evidence_compatibility = []
    final_response = ""
    for notice in notices:
        medication_name = notice["medication_name"]
        contraindication = medication_name + "\n"
        for point in points:
            if point.get("payload").get("medication_name") == medication_name and point.get("payload").get("contraindication"):
                contraindication += point.get("payload").get("text") + " "
        print("contraindication", contraindication)
        final_response, evidence_compatibility = generation(query, contraindication, evidence_compatibility, final_response)
    return final_response, evidence_compatibility
