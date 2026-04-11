from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field

class Observation(BaseModel):
    candidate_skills: List[str] = Field(..., description="Skills of the candidate")
    experience_years: float = Field(..., description="Years of experience")
    expected_salary: int = Field(..., description="Salary candidate expects (in $1000s)")
    educational_background: str = Field(..., description="Highest degree attained")
    soft_skills: List[str] = Field(..., description="Interpersonal and professional soft skills")
    cultural_fit_score: float = Field(..., description="Estimated cultural alignment score (0.0 to 1.0)")
    job_required_skills: List[str] = Field(..., description="Skills required for the job")
    budget_limit: int = Field(..., description="Max salary budget for the role (in $1000s)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and reasoning hints")

class Action(BaseModel):
    decision: Literal["shortlist", "reject", "request_portfolio", "interview"] = Field(..., description="The action to take for this candidate")

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
    score: float = Field(..., description="Standardized performance score (strictly 0.01 to 0.99)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Grading rationale and breakdown")
