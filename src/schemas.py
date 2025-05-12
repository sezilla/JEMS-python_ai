from pydantic import BaseModel, RootModel
from typing import Any, Dict, List, Optional

# Request Body
class TeamAllocationRequest(BaseModel):
    project_id: int
    package_id: int
    start: str
    end: str

class SpecialRequest(BaseModel):
    project_id: int
    special_request: str

class TaskScheduleRequest(BaseModel):
    project_id: int

# Task assignment request

class CheckItem(BaseModel):
    check_item_id: str
    check_item_name: str
    due_date: str
    status: str = 'incomplete'

class Checklist(BaseModel):
    checklist_id: str
    checklist_name: str
    check_items: List[CheckItem]

class CardData(BaseModel):
    card_id: str
    card_name: str
    card_due_date: str
    card_description: str
    checklists: List[Checklist]

class UserData(BaseModel):
    user_id: int
    skills: List[str]

class TaskAllocationRequest(BaseModel):
    project_id: int
    data_array: List[CardData]
    users: Dict[str, List[UserData]]













# Response Body
class TeamAllocationResponse(BaseModel):
    success: bool
    project_id: int
    package_id: int
    start: str
    end: str
    allocated_teams: list[int]

class SpecialRequestResponse(BaseModel):
    success: bool
    project_id: int
    special_request: list[list[str]]

class TaskScheduleResponse(RootModel):
    root: Dict[str, Dict[str, Dict[str, str]]]

class ScheduleResponse(BaseModel):
    success: bool
    project_id: int
    start: str
    end: str
    duration: int
    trello_tasks: TaskScheduleResponse

class ChecklistResponse(BaseModel):
    user_id: int
    check_items: List[CheckItem]

class TaskAllocationResponse(BaseModel):
    success: bool
    project_id: int
    checklists: Dict[str, Dict[str, Any]]
    error: Optional[str] = None
