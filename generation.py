import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generation(treatment: str, contraindication: str, evidence_compatibility: list, final_response: str) -> tuple[str, list[str]]:
    """
    This function takes a medical treatment and a Notice as input 
    and returns a JSON object in French with information about the compatibility of the medications in the treatment 
    and if not compatible, a replacement medication and textual evidence from the Notice.
    The function first calls the OpenAI API with the prompt and the treatment and Notice as input. 
    It then parses the response into a JSON object and extracts the relevant information.
    If the OpenAI API call fails, the function will retry up to 3 times before returning an error message.
    Parameters:
    treatment (str): A medical treatment containing one or more medications.
    contraindication (str): A Notice for a medication.
    Returns:
    tuple[str, list[str]]: A tuple containing the JSON object as a string and a list of textual evidence from the Notice.
    """
    prompt = """You are a medical expert system.
    You are given a Notice for a medication.
    You MUST answer STRICTLY and ONLY using the provided document.
    ### TASK ###
    Given a medical Treatment containing one or more medications:
    1. Determine whether one the medication in Treatment has an explicit incompatibility or contraindication in the Notice and if so, specify them.
    2. If medications are NOT compatible:
    - Determine wether there is a replacement medication suggested in the Notice. Otherwise return empty string.
    3. If there is one or more contraindications in the Notice, return a list of evidence from the document.
    ### IMPORTANT RULES ###
    - Evidences MUST be copied verbatim from the provided documents.
    - Do NOT paraphrase.
    - Do NOT infer.
    - Do NOT combine sentences.
    - If no explicit evidence exists, return an empty string "".
    ### Treatment ###
    """+treatment+"""
    ### Notice ###
    """+contraindication+"""
    ### OUTPUT FORMAT ###
    Return ONLY a valid JSON object in French, with the following structure:
    {
        "compatibility_mentioned": true | false,
        "compatibility_explanation": string,
        "replacement_medication": string | "",
        "evidence_compatibility": [string | ""],
    }
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
                response = {}
            compatibility_explanation = response.get("compatibility_explanation") if response.get("compatibility_mentioned") else "Aucune contre-indication n'est mentionnée."
            replacement_medication = response.get("replacement_medication") or "Aucun médicament de substitution n'est proposé."
            evidence_compatibility.extend(response.get("evidence_compatibility", []) or []) 
            medication_name = contraindication.split("\n")[0]
            final_response += medication_name + "\n" + compatibility_explanation + "\n" + replacement_medication + "\n"
        except Exception as e:
            print(f"Api call failed with error: {e} - attempt {attempt}/{3} - retrying...")
            final_response = "Une erreur s'est produite lors de la tentative de appel de l'API. Veuillez essayer de nouveau."
            evidence_compatibility = []
        return final_response, evidence_compatibility


