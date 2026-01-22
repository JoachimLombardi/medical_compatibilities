import base64
import json
import os
from pathlib import Path
import tempfile
import fitz
from openai import OpenAI


tools = [
    {
        "type": "function",
        "name": "extract_medical_treatment",
        "description": "Extract ALL prescripted medications from a document.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "medical_treatment": {
                    "type": "array",
                    "description": "The list of the prescripted medications, generally preceded by a number and written in capital letters. ex: 1) MEDICATION NAME",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the medication",
                            },
                        },
                        "required": ["name"],
                        "additionalProperties": False
                    },
                },
            },
            "required": ["medical_treatment"],
            "additionalProperties": False
        },
    },
]


def to_b64_image(treatment):
    """
    Convert a treatment to a base64 encoded image.

    This function takes a treatment as an argument (which should be a file-like object)
    and returns a list of base64 encoded images.

    The images are converted from the treatment depending on the file type. If the file type is
    PDF, the images are extracted by rendering the PDF pages as images. If the file type is
    a JPEG or PNG, the image is read directly from the file.

    Args:
        treatment (file-like object): The treatment to convert to an image.

    Returns:
        list[str]: A list of base64 encoded images.
    """
    suffix = "." + treatment.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
     treatment.seek(0)
     tmp_file.write(treatment.read())
     path = Path(tmp_file.name)
    list_b64 = []
    if suffix in [".pdf", ".PDF"]:
        doc = fitz.open(path)
        matrix = fitz.Matrix(2, 2) 
        for page in doc:
            # Convert to image
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("jpeg")
            try:
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
            except Exception as e: 
                print(f"Error: {e}")
                return None
            url= f"data:image/jpeg;base64,{base64_image}"
            list_b64.append(url)
    else:
        with open(path, "rb") as f:
            try:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e: 
                print(f"Error: {e}")
                return None
            mime = "jpeg" if suffix.lower() in [".jpg", ".jpeg"] else "png"
            url= f"data:image/{mime};base64,{base64_image}"
            list_b64.append(url)
    return list_b64


def call_llm(files, tools=tools):
    treatments = ""
    for treatment in files:
        list_images = to_b64_image(treatment)
        messages = [{"role":"user", "content": []}]
        for image_url in list_images:
            messages[0]["content"].append({"type":"input_image", "image_url":image_url})
        data = {
                "model": "gpt-4.1",
                "input": messages,
                "tools": tools,
                "tool_choice": {"type": tools[0]["type"], "name": tools[0]["name"]},
                "temperature": 0,
                }
        for attempt in range(1,4):
            try:
                print("api gpt call")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.responses.create(**data)
                final_response = response.output[0].to_json()
                print("final_response", final_response)
                if isinstance(final_response, dict):
                    treatment_dict = final_response.get("arguments", None)
                    treatment_dict = json.loads(treatment_dict)
                elif isinstance(final_response, str):
                    treatment_dict = json.loads(final_response)
                    treatment_dict = treatment_dict.get("arguments", None)
                    treatment_dict = json.loads(treatment_dict)
                else:
                    raise TypeError(f"final_response is not a string or a dict, it's a {type(final_response)}")
                for medication in treatment_dict.get("medical_treatment", []):
                    treatments += "\n" + medication.get("name", "")
                break
            except Exception as e:
                print(f"Attempt {attempt}/3 \n API call failed with error: {e} - retrying...")
    return treatments


