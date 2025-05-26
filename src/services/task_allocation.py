import os
import re
import sys
import json
import traceback
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.database import SessionLocal
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.schemas import CardData, CheckItem, Checklist, TaskAllocationRequest, TaskAllocationResponse, UserData

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
TASK_ALLOCATION_HISTORY = os.path.join(history_dir, "task_allocation.json")

os.makedirs(history_dir, exist_ok=True)

if not os.path.exists(TASK_ALLOCATION_HISTORY):
    with open(TASK_ALLOCATION_HISTORY, "w") as f:
        json.dump([], f)

def extract_json_from_text(text: str) -> str:
    """Enhanced JSON extraction with multiple strategies"""
    if not text or not text.strip():
        raise ValueError("Empty response text")
    
    # Strategy 1: Find complete JSON object
    brace_count = 0
    start_pos = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_pos == -1:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                return text[start_pos:i+1]
    
    # Strategy 2: Find between first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text[start:end+1]
        try:
            json.loads(potential_json)
            return potential_json
        except:
            pass
    
    # Strategy 3: Look for JSON code blocks
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'`(\{.*?\})`'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
    
    # Strategy 4: Clean common formatting issues
    cleaned = text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    
    raise ValueError("No valid JSON found in response")

def fix_json_formatting(json_str: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix unescaped quotes in strings
    json_str = re.sub(r'(?<!\\)"(?=.*".*:)', r'\\"', json_str)
    
    # Ensure proper string quoting for keys
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Fix already quoted keys (avoid double quotes)
    json_str = re.sub(r'""(\w+)"":', r'"\1":', json_str)
    
    return json_str

def validate_and_parse_json(text: str) -> dict:
    """Robust JSON parsing with multiple attempts"""
    original_text = text
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract JSON from text
    try:
        json_str = extract_json_from_text(text)
    except ValueError as e:
        raise ValueError(f"Could not extract JSON: {e}")
    
    # Try parsing extracted JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Try fixing formatting issues
    try:
        fixed_json = fix_json_formatting(json_str)
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to repair JSON structure
    try:
        return repair_json_structure(json_str)
    except Exception:
        raise ValueError(f"Could not parse JSON after all attempts. Original: {original_text[:200]}...")

def repair_json_structure(json_str: str) -> dict:
    """Attempt to repair malformed JSON structure"""
    # Remove any text before first {
    start = json_str.find('{')
    if start > 0:
        json_str = json_str[start:]
    
    # Remove any text after last }
    end = json_str.rfind('}')
    if end != -1:
        json_str = json_str[:end+1]
    
    # Try to balance braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        # Remove extra closing braces from the end
        diff = close_braces - open_braces
        for _ in range(diff):
            last_brace = json_str.rfind('}')
            if last_brace != -1:
                json_str = json_str[:last_brace] + json_str[last_brace+1:]
    
    return json.loads(json_str)

def validate_allocation_structure(data: dict) -> dict:
    """Validate and fix the allocation response structure"""
    if not isinstance(data, dict):
        raise ValueError("Response must be a dictionary")
    
    # Auto-fix: If the response is missing the "checklists" wrapper but contains checklist data
    if "checklists" not in data:
        # Check if the root object contains checklist-like data
        if all(isinstance(v, dict) and "checklist_name" in v and "check_items" in v for v in data.values()):
            print("Auto-fixing: Adding missing 'checklists' wrapper")
            data = {"checklists": data}
        else:
            raise ValueError("Response missing 'checklists' key and doesn't contain valid checklist data")
    
    checklists = data["checklists"]
    if not isinstance(checklists, dict):
        raise ValueError("'checklists' must be a dictionary")
    
    for checklist_id, checklist_data in checklists.items():
        if not isinstance(checklist_data, dict):
            raise ValueError(f"Checklist {checklist_id} must be a dictionary")
        
        # Ensure required fields exist
        if "checklist_name" not in checklist_data:
            checklist_data["checklist_name"] = f"Checklist {checklist_id}"
        
        if "check_items" not in checklist_data:
            checklist_data["check_items"] = []
        
        if not isinstance(checklist_data["check_items"], list):
            raise ValueError(f"Checklist {checklist_id} 'check_items' must be a list")
        
        # Validate each check item
        for i, item in enumerate(checklist_data["check_items"]):
            if not isinstance(item, dict):
                raise ValueError(f"Check item {i} in checklist {checklist_id} must be a dictionary")
            
            # Ensure required fields
            required_fields = ["check_item_id", "check_item_name", "due_date", "status", "user_id", "priority"]
            for field in required_fields:
                if field not in item:
                    if field == "status":
                        item[field] = "incomplete"
                    elif field == "priority":
                        item[field] = "p1"
                    elif field == "user_id":
                        item[field] = 0  # Will need to be assigned
                    else:
                        item[field] = f"default_{field}"
    
    return data

def create_system_prompt() -> str:
    """Create a comprehensive system prompt"""
    return """You are a task allocation AI. Your job is to assign tasks to users based on their skills and department.

CRITICAL RULES:
1. Each card represents a department
2. Only assign users from the matching department to that card's tasks
3. Match users to tasks based on their skills
4. Distribute tasks fairly among users
5. Assign priority levels: p0 (highest), p1 (medium), p2 (lowest)
6. MUST return ONLY valid JSON - no explanations, no markdown, no additional text

REQUIRED JSON FORMAT - FOLLOW EXACTLY:
{
  "checklists": {
    "checklist_id_1": {
      "checklist_name": "string",
      "check_items": [
        {
          "check_item_id": "string",
          "check_item_name": "string", 
          "due_date": "string",
          "status": "incomplete",
          "user_id": 123,
          "priority": "p0|p1|p2"
        }
      ]
    }
  }
}

CRITICAL - THE RESPONSE MUST:
- Start with {"checklists": {
- Have the "checklists" key as the root property
- Use exact checklist_id values as keys inside "checklists"
- End with }}
- No trailing commas anywhere
- user_id must be an integer from the provided users
- priority must be exactly: p0, p1, or p2

EXAMPLE STRUCTURE:
{"checklists": {"abc123": {"checklist_name": "Test", "check_items": [{"check_item_id": "xyz", "check_item_name": "Task", "due_date": "2024-01-01", "status": "incomplete", "user_id": 123, "priority": "p1"}]}}}"""

def create_user_prompt(project_id: int, task_summary: List[dict], users_dict: Dict[str, List[dict]]) -> str:
    """Create the user prompt with data"""
    return f"""Project ID: {project_id}

TASKS TO ALLOCATE:
{json.dumps(task_summary, indent=2)}

AVAILABLE USERS BY DEPARTMENT:
{json.dumps(users_dict, indent=2)}

CRITICAL: Your response must be EXACTLY in this format:
{{"checklists": {{
  "checklist_id": {{
    "checklist_name": "name",
    "check_items": [...]
  }}
}}}}

Assign each task to the most suitable user from the correct department based on skills. Ensure fair distribution and appropriate priority levels.

Remember: The response MUST start with {{"checklists": and contain all checklist IDs as keys within the checklists object."""

def allocate_tasks(request: TaskAllocationRequest) -> TaskAllocationResponse:
    """Main task allocation function with enhanced error handling"""
    try:
        project_id = request.project_id
        data_array = request.data_array
        users_by_department = request.users

        # Build task summary
        task_summary = []
        for card in data_array:
            for checklist in card.checklists:
                for item in checklist.check_items:
                    task_summary.append({
                        "card_id": card.card_id,
                        "card_name": card.card_name,
                        "card_due_date": card.card_due_date,
                        "card_description": card.card_description,
                        "checklist_id": checklist.checklist_id,
                        "checklist_name": checklist.checklist_name,
                        "check_item_id": item.check_item_id,
                        "check_item_name": item.check_item_name,
                        "due_date": item.due_date,
                        "status": item.status
                    })

        # Convert users to dict format
        users_dict = {
            dept: [user.model_dump() for user in user_list]
            for dept, user_list in users_by_department.items()
        }

        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(project_id, task_summary, users_dict)

        max_attempts = 5  # Increased attempts
        last_error = None
        raw_responses = []

        for attempt in range(max_attempts):
            try:
                print(f"Attempt {attempt + 1}/{max_attempts}")
                
                # Adjust temperature based on attempt (start conservative)
                temperature = 0.3 + (attempt * 0.1)
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": '{"checklists": {'},
                    ],
                    max_tokens=4000,
                    temperature=min(temperature, 0.7)
                )

                response_text = response.choices[0].message.content.strip()
                
                # Auto-fix: If the response doesn't start with the expected format, prepend it
                if not response_text.startswith('{"checklists":'):
                    if response_text.startswith('{'):
                        # The model started with a checklist ID directly
                        response_text = '{"checklists": ' + response_text + '}'
                        print("Auto-fixed: Added checklists wrapper to response")
                    else:
                        response_text = '{"checklists": {' + response_text + '}}'
                        print("Auto-fixed: Added complete checklists wrapper")
                
                raw_responses.append(response_text)
                
                print(f"Raw response length: {len(response_text)}")
                print(f"Raw response preview: {response_text[:200]}...")

                # Parse and validate JSON
                parsed_data = validate_and_parse_json(response_text)
                validated_data = validate_allocation_structure(parsed_data)

                print(f"Successfully parsed and validated response on attempt {attempt + 1}")
                
                return TaskAllocationResponse(
                    success=True,
                    project_id=request.project_id,
                    checklists=validated_data["checklists"]
                )

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Log the specific error type
                if "JSON" in str(e).upper():
                    print(f"JSON parsing error on attempt {attempt + 1}")
                elif "validation" in str(e).lower():
                    print(f"Validation error on attempt {attempt + 1}")
                else:
                    print(f"Other error on attempt {attempt + 1}: {type(e).__name__}")

        # All attempts failed - create fallback response
        print(f"All {max_attempts} attempts failed. Creating fallback response.")
        fallback_response = create_fallback_allocation(request)
        if fallback_response:
            return fallback_response

        # Last resort error response
        return TaskAllocationResponse(
            success=False,
            project_id=request.project_id,
            checklists={},
            error=f"Failed after {max_attempts} attempts. Last error: {str(last_error)}. Raw responses available for debugging."
        )

    except Exception as e:
        logging.error(f"Critical error in task allocation: {e}")
        traceback.print_exc()
        return TaskAllocationResponse(
            success=False,
            project_id=request.project_id,
            checklists={},
            error=f"Critical system error: {str(e)}"
        )

def create_fallback_allocation(request: TaskAllocationRequest) -> Optional[TaskAllocationResponse]:
    """Create a basic fallback allocation when AI fails"""
    try:
        print("Creating fallback allocation...")
        
        checklists = {}
        
        # Get all users flattened
        all_users = []
        for dept, users in request.users.items():
            all_users.extend(users)
        
        if not all_users:
            return None
        
        user_index = 0
        
        for card in request.data_array:
            for checklist in card.checklists:
                checklist_data = {
                    "checklist_name": checklist.checklist_name,
                    "check_items": []
                }
                
                for item in checklist.check_items:
                    # Round-robin assignment
                    assigned_user = all_users[user_index % len(all_users)]
                    user_index += 1
                    
                    check_item_data = {
                        "check_item_id": item.check_item_id,
                        "check_item_name": item.check_item_name,
                        "due_date": item.due_date,
                        "status": item.status,
                        "user_id": assigned_user.user_id,
                        "priority": "p1"  # Default priority
                    }
                    
                    checklist_data["check_items"].append(check_item_data)
                
                checklists[checklist.checklist_id] = checklist_data
        
        print(f"Fallback allocation created with {len(checklists)} checklists")
        
        return TaskAllocationResponse(
            success=True,
            project_id=request.project_id,
            checklists=checklists
        )
        
    except Exception as e:
        print(f"Fallback allocation failed: {e}")
        return None

# Test code remains the same
if __name__ == "__main__":
    from pydantic import BaseModel
    import uuid

    def uid():
        return str(uuid.uuid4())[:24]

    dummy_request = TaskAllocationRequest(
        project_id=1,
        data_array=[
            CardData(
                card_id=uid(),
                card_name="Design",
                card_due_date="2024-03-30",
                card_description="Design phase tasks",
                checklists=[
                    Checklist(
                        checklist_id="67ff892b923c801dfa8cf7e8",
                        checklist_name="Design Checklist",
                        check_items=[
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cf9ff",
                                check_item_name="Create mockups",
                                due_date="2024-03-25"
                            ),
                        ]
                    )
                ]
            ),
            CardData(
                card_id=uid(),
                card_name="Development",
                card_due_date="2024-03-31",
                card_description="Development phase tasks",
                checklists=[
                    Checklist(
                        checklist_id="67ff892b923c801dfa8cf7eb",
                        checklist_name="Development Checklist",
                        check_items=[
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cfa0c",
                                check_item_name="Build login API",
                                due_date="2024-03-26"
                            ),
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cfa0d",
                                check_item_name="Setup database schema",
                                due_date="2024-03-27"
                            )
                        ]
                    )
                ]
            )
        ],
        users={
            "Design": [
                UserData(user_id=188, skills=["Figma", "UX", "Wireframes"]),
                UserData(user_id=194, skills=["Branding", "Illustrator"])
            ],
            "Development": [
                UserData(user_id=250, skills=["Laravel", "MySQL"]),
                UserData(user_id=194, skills=["API", "PHP", "Databases"])
            ]
        }
    )

    try:
        schedule = allocate_tasks(dummy_request)
        print("\n=== Allocation Result ===")
        print(json.dumps(schedule.model_dump(), indent=4))
    except Exception as e:
        print(f"Error generating schedule: {e}")