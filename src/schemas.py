from pydantic import BaseModel

class ProjectAllocationRequest(BaseModel):
    project_id: int
    package_id: int
    start: str
    end: str
