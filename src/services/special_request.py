import os
import sys
import json
from openai import OpenAI
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.database import SessionLocal
from src.models import Department
from src.schemas import SpecialRequest, SpecialRequestResponse

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
HISTORY_SPECIAL_REQUEST = os.path.join(history_dir, "special_requests.json")

os.makedirs(history_dir, exist_ok=True)

if not os.path.exists(HISTORY_SPECIAL_REQUEST):
    with open(HISTORY_SPECIAL_REQUEST, "w") as f:
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

def generate_special_request(special_request: SpecialRequest) -> SpecialRequestResponse:
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
        ["<department_name>", "<task_description>"]
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
        parsed_response = json.loads(cleaned_json)
        return SpecialRequestResponse(**parsed_response)
    except (json.JSONDecodeError, AttributeError, IndexError, ValidationError) as e:
        raise ValueError(f"Failed to parse JSON from AI response: {output_text}") from e

def save_special_request(special_request: SpecialRequestResponse):
    """Saves special requests history in JSON file."""
    try:
        with open(HISTORY_SPECIAL_REQUEST, "r+") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

            history.append(special_request.dict())
            f.seek(0)
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving special request: {e}")

if __name__ == "__main__":
    test_request = SpecialRequest(
        project_id=1,
        special_request="""
        For our wedding, we would like to incorporate a few special elements to make the day truly unique. 
        We request a floral arrangement featuring white and blush roses with eucalyptus accents for the ceremony and reception. 
        Our color theme is navy blue and gold, so we’d love for the décor, table settings, and lighting to complement this palette. 
        For the ceremony, we would like a live string quartet to play as guests arrive and during the vows. During the reception, 
        we’d prefer a mix of classic jazz and modern hits for the background music, with a DJ taking over for the dance portion. 
        Additionally, we request a fully vegetarian menu with gluten-free options for select guests. For dessert, 
        we’d love a tiered cake with vanilla and raspberry flavors, along with a small dessert bar featuring macarons and chocolate-covered strawberries. 
        We also plan to have a sparkler send-off at the end of the night, so we’d appreciate assistance in organizing this safely.
        """
    )

    try:
        special_request_response = generate_special_request(test_request)
        save_special_request(special_request_response)
        print("Generated Special Request:")
        print(json.dumps(special_request_response.dict(), indent=4))
    except Exception as e:
        print(f"Error generating special request tasks: {e}")
