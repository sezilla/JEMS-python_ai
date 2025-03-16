from pydantic import BaseModel

# Request Body
class ProjectAllocationRequest(BaseModel):
    project_id: int
    package_id: int
    start: str
    end: str

class SpecialRequest(BaseModel):
    project_id: int
    special_request: str

class CategoryScheduleRequest(BaseModel):
    project_id: int
    start: str
    end: str


# Response Body
class TeamAllocationResponse(BaseModel):
    success: bool
    project_id: int
    package_id: int
    start: str
    end: str

class SpecialRequestResponse(BaseModel):
    success: bool
    project_id: int
    special_request: list[list[str]]

class CategoryScheduleResponse(BaseModel):
    success: bool
    project_id: int
    start: str
    end: str
    duration: int
    categories: list[list[str]]