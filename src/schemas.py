from pydantic import BaseModel

class ProjectAllocationRequest(BaseModel):
    project_id: int
    package_id: int
    start: str
    end: str

class ProjectSpecialRequest(BaseModel):
    project_id: int
    special: str

class CategoryScheduleRequest(BaseModel):
    project_id: int
    start: str
    end: str