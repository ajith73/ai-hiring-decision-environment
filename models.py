from typing import List, Literal
from pydantic import BaseModel, Field

class Observation(BaseModel):
    candidate_skills: List[str] = Field(..., description="Skills of the candidate")
    experience_years: int = Field(..., description="Years of experience")
    job_required_skills: List[str] = Field(..., description="Skills required for the job")

class Action(BaseModel):
    decision: Literal["shortlist", "reject"] = Field(..., description="Decision for the candidate")

class Reward(BaseModel):
    score: float = Field(..., description="Reward score")

class Task(BaseModel):
    task_id: int
    description: str
    expected_decision: str

class GradeRequest(BaseModel):
    task_id: int
    action: str

class GradeResponse(BaseModel):
    score: float
