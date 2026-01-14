import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generation(query: str, adverse_effects: str, contraindication: str) -> str:
    """
    Generate a medical response based on a query and two documents.
    The two documents contain information about adverse effects and contraindications of medications.
    The response is a JSON object in French, with the following structure:
    {
        "adverse_effects": [{
            "<medication_name>": {
            "effects": [string],
            "evidence": [string]
            }]
        },
        "compatibility_mentioned": true | false,
        "compatibility_explainaition": string,
        "replacement_medication": string | "",
        "evidence_compatibility": string
    }
    """
    prompt = """You are a medical expert system.
    You MUST answer STRICTLY and ONLY using the provided documents.
    You are NOT allowed to use external knowledge.
    If information is missing, you MUST return an empty string or list.
    ### TASK ###
    Given a medical Treatment containing one or more medications:
    1. For EACH medication, list ALL adverse effects explicitly mentioned in the Adverse Effects document.
    2. Determine whether the medications in the Treatment are compatible.
    - Medications are NOT compatible ONLY IF an explicit incompatibility or contraindication between them is stated in the Contraindication document.
    - If not explicitly mentioned in the Contraidication document compatibility_mentioned must be false
    3. If medications are NOT compatible:
    - Identify a replacement medication ONLY IF explicitly mentioned in the Contraindication document.
    - Otherwise return empty string.
    4. For EVERY list of adverse effects or compatibility decision, provide an EXACT textual evidence.
    ### IMPORTANT RULES ###
    - Evidence MUST be copied verbatim from the provided documents.
    - Do NOT paraphrase.
    - Do NOT infer.
    - Do NOT combine sentences.
    - If no explicit evidence exists, return an empty string "".
    - If compatibility cannot be determined, return false.
    ### Treatment ###
    {{TREATMENT}}
    ### Adverse Effects Document ###
    {{ADVERSE_EFFECTS}}
    ### Contraindication Document ###
    {{CONTRAINDICATION}}
    ### OUTPUT FORMAT ###
    Return ONLY a valid JSON object in French, with the following structure:
    {
    "adverse_effects": [{
        "<medication_name>": {
        "effects": [string],
        "evidence": string
        }]
    },
    "compatibility_mentioned": true | false,
    "compatibility_explainaition": string,
    "replacement_medication": string | "",
    "evidence_compatibility": string
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
                response = {"compatibility_mentioned": False, "explanation": "Not enough information in the document."}
            evidence = ""
            if ("adverse_effects" in response and "compatibility_mentioned" in response and "compatibility_explanation" in response 
                and "replacement_medication" in response and "evidence_compatibility" in response):
                adverse_effects = response.get("adverse_effects")
                compatibility_explanation = response.get("compatibility_explanation")
                replacement_medication = response.get("replacement_medication", "")
                evidence_compatibility = response.get("evidence_compatibility")
                for medication in adverse_effects:
                    for effect in medication.get("effects"):
                        evidence += medication.get("evidence")
                        evidence += "\n"
                evidence += compatibility_explanation
                evidence += "\n"
                evidence += evidence_compatibility
                evidence += "\n"
                evidence += "Remplacement : " + replacement_medication
                final_response = response.get('explanation')
            else:
                final_response = "Cette requête n'a pas pu aboutir. Veuillez nous excuser."
                retrieval = ""
            return final_response, evidence, retrieval
        except Exception as e:
            print(f"Api call failed with error: {e} - attempt {attempt + 1}/{3} - retrying...")
