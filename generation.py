import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generation(treatment: str, adverse_effects: str, contraindication: str) -> str:
    """
    Generate a medical report based on the given treatment, adverse effects and contraindication documents.

    The function takes in the treatment, adverse effects and contraindication documents as strings.
    It returns a json object with the following structure:
    {
        "adverse_effects": [{
            "medication_name",
            "evidence": string
        }],
        "compatibility_mentioned": true | false,
        "compatibility_explaination": string,
        "replacement_medication": string | "",
        "evidence_compatibility": [{
                "medication_name": string,
                "evidence": string
            }
        ] | "",
    }

    The function uses the OpenAI API to generate the report based on the given documents.
    If the API call fails, the function will retry up to 3 times before returning an error message.

    Parameters:
    treatment (str): The treatment document as a string.
    adverse_effects (str): The adverse effects document as a string.
    contraindication (str): The contraindication document as a string.

    Returns:
    tuple[str, dict[str, str], dict[str, str]]: A tuple containing the generated report as a string, the adverse effects as a dictionary and the evidence for compatibility as a dictionary.
    """
    prompt = """You are a medical expert system.
    You MUST answer STRICTLY and ONLY using the provided documents.
    You are NOT allowed to use external knowledge.
    If information is missing, you MUST return an empty string.
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
    """+treatment+"""
    ### Adverse Effects Document ###
    Each document has the following format:
    [TITLE]: <medication name>
    [CONTENT]: <document text>
    """+adverse_effects+"""
    ### Contraindication Document ###
    Each document has the following format:
    [TITLE]: <medication name>
    [CONTENT]: <document text>
    """+contraindication+"""
    ### OUTPUT FORMAT ###
    Return ONLY a valid JSON object in French, with the following structure:
    {
        "adverse_effects": [{
            "medication_name",
            "evidence": string
        }],
        "compatibility_mentioned": true | false,
        "compatibility_explainaition": string,
        "replacement_medication": string | "",
        "evidence_compatibility": [{
                "medication_name": string,
                "evidence": string
            }
        ] | "",
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
            final_response = ""
            if ("adverse_effects" in response and "compatibility_mentioned" in response and "compatibility_explanation" in response 
                and "replacement_medication" in response and "evidence_compatibility" in response):
                adverse_effects = response.get("adverse_effects")
                compatibility_explanation = response.get("compatibility_explanation")
                replacement_medication = response.get("replacement_medication", "")
                evidence_compatibility = response.get("evidence_compatibility")
                for medication in adverse_effects:
                        medication_name = medication.get("medication_name")
                        effects = medication.get("evidence")
                        final_response += "Effets indésirables de " + medication_name + " : " + effects
                        final_response += "\n"
                final_response += compatibility_explanation
                final_response += "\n"
                final_response += "Remplacement : " + replacement_medication
            else:
                final_response = "Cette requête n'a pas pu aboutir. Veuillez nous excuser."
            return final_response, adverse_effects, evidence_compatibility
        except Exception as e:
            print(f"Api call failed with error: {e} - attempt {attempt + 1}/{3} - retrying...")
