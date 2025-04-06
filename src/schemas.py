from pydantic import BaseModel, RootModel
from typing import Dict, List, Optional

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
