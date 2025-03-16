import os
import sys
import json
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.database import SessionLocal
from src.models import Department 
from src.schemas import SpecialRequest

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
history_file = os.path.join(history_dir, "special_requests.json")

os.makedirs(history_dir, exist_ok=True)

if not os.path.exists(history_file):
    with open(history_file, "w") as f:
        json.dump([], f)

def get_departments():
    session = SessionLocal()
    try:
        departments = session.query(Department).all()
        return departments
    finally:
        session.close()

def clean_json_response(response_text: str) -> str:
    """Cleans up AI-generated JSON response safely."""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()

def generate_special_request(special_request: SpecialRequest) -> dict:
    departments = get_departments()
    
    department_data = "\n".join(
        [f"- {dept.name}: {dept.description}" for dept in departments]
    )

    prompt = f"""
You are an AI assistant for special requests. Follow these rules:
    - Generate task(s) based on the given special request.
    - Assign generated tasks to the appropriate department based on its description.
    - Ensure each task is clear, concise, and includes all necessary details.
    - Multiple tasks may be created and assigned to multiple departments.
    - A department may receive multiple tasks.
    
Special Request Details:
    Project ID: {special_request.project_id}
    Request Description: {special_request.special_request}

Available Departments:
{department_data}

Please generate a structured JSON output in the format:
{{
    "success": <bool>,
    "project_id": {special_request.project_id},
    "special_request": [
        {{
            "department": "<department_name>",
            "task": "<task_description>"
        }},
        ...
    ]
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a structured task generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=2000,
        top_p=1
    )

    try:
        output_text = response.choices[0].message.content
        cleaned_json = clean_json_response(output_text)
        return json.loads(cleaned_json)
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        raise ValueError(f"Failed to parse JSON from AI response: {output_text}") from e

def save_special_request(special_request: dict):
    """Saves special requests history in JSON file."""
    try:
        with open(history_file, "r+") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

            history.append(special_request)
            f.seek(0)
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving special request: {e}")

if __name__ == "__main__":
    test_project_id = 1
    test_request = """The couple envisions a **fairytale-themed wedding**, where every detail creates an enchanting experience. 
    The event will feature a regal banquet with gourmet dishes, a molecular gastronomy station, and signature themed cocktails. 
    The bride will have a princess-inspired look, while the groom and bridal party embrace a polished, royal style. 
    Cinematic storytelling will capture magical moments with drone and slow-motion shots. 
    The venue will be transformed with cascading floral arches, chandeliers, and a candlelit aisle. 
    A live orchestra, harpist, and a surprise royal ball dance will add to the charm, with seamless coordination ensuring a flawless and stress-free celebration.
    """

    try:
        special_request = generate_special_request(test_project_id, test_request)
        save_special_request(special_request)
        print("Generated Special Request:")
        print(json.dumps(special_request, indent=4))
    except Exception as e:
        print(f"Error generating special request tasks: {e}")