import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generation(query: str, medication_1: str, medication_2: str, retrieval: str) -> str:
    """
    Generate a medical compatibility response using the provided document.
    Parameters:
        query (str): The query to ask about the medications or condition.
        retrieval (QueryResponse): The retrieved document.
    Returns:
        str: A valid JSON object containing the compatibility and explanation.
    The response should be in the following format:
    {
        "compatibility": true | false | null,
        "explanation": string
    }
    If the document does not contain enough information, the response should be:
    {"compatibility": null, "explanation": "Not enough information in the document."}
    """
    prompt = """You are a medical expert system.
    You must answer STRICTLY using the provided document.
    If the document does not contain enough information, answer:
    {"compatibility": null, "explanation": "Not enough information in the document."}
    ### Task ###
    Determine whether the two medications or the medication and condition are compatible.
    ### Input ###
    """+query+"""
    ### Document ###
    """+retrieval+"""
    ### Output ###
    {
        "compatibility": true | false | null,
        "explanation": string,
        "evidence": string 
    }
    Rules for "evidence":
    - It MUST be an EXACT QUOTE copied verbatim from the Document
    - Do NOT paraphrase
    - Do NOT add any text not present in the Document
    - If no explicit evidence exists, return empty string
    Return ONLY a valid JSON object, no extra text.
    """
    messages = [{"role": "user", "content": prompt}]
    data = {
        "model": "gpt-4.1",
        "input": messages,
        "temperature": 0
    }
    for attempt in range(1,4):
        try:
            print("api gpt call")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.responses.create(**data)
            response = response.output_text
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {"compatibility": None, "explanation": "Not enough information in the document."}
            if "compatibility" in response and "explanation" and "evidence" in response:
                compatibility = response.get("compatibility")
                if compatibility:
                    final_response = f"{medication_1} est compatible avec {medication_2}. \n {response.get('explanation')}" 
                    evidence = response.get("evidence")
                elif compatibility is None:
                    final_response = f"Les documents ne permettent pas de déterminer la compatibilité entre {medication_1} et {medication_2}." 
                    evidence = ""
                    retrieval = ""
                else:
                    final_response = f"{medication_1} n'est pas compatible avec {medication_2}. \n {response.get('explanation')}"
                    evidence = response.get("evidence")
                    print(evidence)
            else:
                final_response = "Cette requête n'a pas pu aboutir. Veuillez nous excuser."
                evidence = ""
                retrieval = ""
            print(final_response)
            return final_response, evidence, retrieval
        except Exception as e:
            print(f"Api call failed with error: {e} - attempt {attempt + 1}/{3} - retrying...")
