import random
import os
import sys
from fastapi import FastAPI, HTTPException
from typing import List, Literal, Dict, Any

# Ensure the root directory is in the path for models import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import Observation, Action, Reward, Task, GradeRequest, GradeResponse
except ImportError:
    # Fallback for different execution contexts
    from ..models import Observation, Action, Reward, Task, GradeRequest, GradeResponse

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Startup debug log for platform verification."""
    print("App started successfully")

@app.get("/")
def root():
    """Root endpoint for status check and landing page."""
    return {
        "project": "AI Hiring Decision Environment (OpenEnv)",
        "description": "Simulates recruiter decision-making for candidate screening.",
        "status": "running"
    }

@app.get("/health")
def health():
    """Health check endpoint for platform monitoring."""
    return {"status": "ok", "uptime_check": True}

class HiringEnv:
    """
    Simulation environment for evaluating AI agents in a recruitment context.
    Encapsulates candidate data, job requirements, and evaluation logic.
    """
    def __init__(self):
        # Simulation Dataset: A set of candidates for general interaction
        self.simulation_pool = [
            {"id": 1, "skills": ["React", "Node.js"], "experience_years": 3},
            {"id": 2, "skills": ["React", "Python"], "experience_years": 1},
            {"id": 3, "skills": ["Java", "SQL"], "experience_years": 10},
            {"id": 4, "skills": ["React", "Node.js", "SQL"], "experience_years": 5},
            {"id": 5, "skills": ["Node.js"], "experience_years": 2}
        ]
        
        # Target Job Specification
        self.job_specification = {
            "required_skills": ["React", "Node.js"],
            "min_experience": 2
        }
        
        # Evaluation Tasks: Realistic recruitment scenarios (Easy -> Medium -> Hard)
        self.evaluation_tasks = [
            {
                "task_id": 1, 
                "description": "Senior Fullstack Developer with 5 years experience (React, Node.js) applying for a lead role.", 
                "expected": "shortlist",
                "candidate": {"skills": ["React", "Node.js"], "experience_years": 5}
            },
            {
                "task_id": 2, 
                "description": "Frontend Developer with 3 years experience (React) applying for a fullstack role needing Node.js.", 
                "expected": "shortlist",
                "candidate": {"skills": ["React"], "experience_years": 3}
            },
            {
                "task_id": 3, 
                "description": "Junior Backend Developer with 1 year experience (Python only) applying for a senior React/Node.js position.", 
                "expected": "reject",
                "candidate": {"skills": ["Python"], "experience_years": 1}
            }
        ]
        
        # Internal State Management
        self.active_candidate = None
        self.current_steps = 0
        self.episode_done = False

    def reset(self, task_id: int = None) -> Observation:
        """
        Resets the session. If task_id is specified, loads that scenario.
        Otherwise, picks a random candidate from the simulation pool.
        """
        if task_id is not None:
            task = next((t for t in self.evaluation_tasks if t["task_id"] == task_id), None)
            if not task:
                raise HTTPException(status_code=404, detail="Task scenario not found")
            self.active_candidate = task["candidate"]
        else:
            self.active_candidate = random.choice(self.simulation_pool)
            
        self.current_steps = 0
        self.episode_done = False
        return self._generate_observation()

    def step(self, action: Action) -> tuple:
        """
        Processes an agent's decision and calculates the reward based on recruitment logic.
        """
        self.current_steps += 1
        
        # Episode safety: Limit total steps to prevent runaway loops
        if self.current_steps > 5:
            self.episode_done = True
            return self._generate_observation(), -0.5, True, {"info": "Evaluation timeout: Step limit exceeded"}
            
        if self.episode_done:
            return self._generate_observation(), 0.0, True, {"info": "Episode already concluded."}
        
        # Decision Logic: Check skills and experience alignment
        cand_skills = set(self.active_candidate["skills"])
        req_skills = set(self.job_specification["required_skills"])
        expr = self.active_candidate["experience_years"]
        min_expr = self.job_specification["min_experience"]
        
        has_full_skills = req_skills.issubset(cand_skills)
        has_min_exp = expr >= min_expr
        has_partial_match = bool(req_skills & cand_skills)
        
        base_reward = 0.0
        
        if has_full_skills and has_min_exp:
            # Ideal candidate: High reward for shortlisting
            base_reward = 1.0 if action.decision == "shortlist" else -1.0
        elif has_partial_match:
            # Borderline candidate: Reward conservative shortlisting
            base_reward = 0.5 if action.decision == "shortlist" else -0.5
        else:
            # Poor match: Reward rejection
            base_reward = -1.0 if action.decision == "shortlist" else 0.5
        
        # Efficiency Penalty: Encourage faster decision-making
        final_reward = base_reward - (0.1 * self.current_steps)
        
        self.episode_done = True
        return self._generate_observation(), final_reward, True, {
            "evaluation_note": "Decision processed", 
            "decision_was": action.decision,
            "steps_taken": self.current_steps
        }

    def grade_decision(self, task_id: int, action: str) -> float:
        """
        Deterministic grader used to benchmark agent performance on specific tasks.
        """
        task = next((t for t in self.evaluation_tasks if t["task_id"] == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Task scenario not found")
        
        target = task["expected"]
        
        if action == target:
            return 1.0
        elif action == "shortlist" and target == "reject":
            # Significant failure: Shortlisting an unqualified candidate
            return 0.0
        else:
            # Minor failure: Erring on the side of caution or missed opportunity
            return 0.5

    def get_current_state_summary(self) -> Dict[str, Any]:
        """Provides a telemetry summary of the current environment state."""
        return {
            "candidate_dataset": self.active_candidate,
            "session_steps": self.current_steps,
            "is_complete": self.episode_done,
            "target_specification": self.job_specification
        }
    
    def _generate_observation(self) -> Observation:
        """Helper to construct the observation payload for the agent."""
        if not self.active_candidate:
            return Observation(candidate_skills=[], experience_years=0, job_required_skills=[])
        return Observation(
            candidate_skills=self.active_candidate["skills"],
            experience_years=self.active_candidate["experience_years"],
            job_required_skills=self.job_specification["required_skills"]
        )

# Global environment instance
recruitment_env = HiringEnv()

@app.post("/reset", response_model=Observation)
async def reset(task_id: int = None):
    """Resets the recruitment session, optionally for a specific task scenario."""
    return recruitment_env.reset(task_id)

@app.post("/step")
async def step(action: Action):
    """Submits a recruitment decision (shortlist/reject) for the current candidate."""
    obs, reward, done, info = recruitment_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/tasks")
async def get_tasks():
    """Returns the official evaluation task suite and interaction schema."""
    return {
        "tasks": [
            {
                "task_id": t["task_id"], 
                "description": t["description"], 
                "expected_decision": t["expected"]
            } for t in recruitment_env.evaluation_tasks
        ],
        "action_schema": {
            "decision": ["shortlist", "reject"],
            "type": "Literal"
        }
    }

@app.post("/grader", response_model=GradeResponse)
async def grader(request: GradeRequest):
    """Benchmarks a specific task action against expert grading criteria."""
    score = recruitment_env.grade_decision(request.task_id, request.action)
    return GradeResponse(score=score)

@app.get("/baseline")
async def run_baseline_benchmarks():
    """Runs an automated baseline agent across all defined tasks to establish a performance floor."""
    benchmarks = []
    accumulated_points = 0.0
    
    for scenario in recruitment_env.evaluation_tasks:
        tid = scenario["task_id"]
        # Initialize scenario
        obs = recruitment_env.reset(tid)
        
        # Baseline Logic: A simple heuristic based on skill overlap
        overlap = bool(set(obs.job_required_skills) & set(obs.candidate_skills))
        logical_decision = "shortlist" if overlap else "reject"
            
        # Evaluation
        points = recruitment_env.grade_decision(tid, logical_decision)
        
        benchmarks.append({
            "task_id": tid,
            "heuristic_decision": logical_decision,
            "points_earned": points
        })
        accumulated_points += points
        
    return {
        "benchmarks": benchmarks,
        "aggregate_performance": (accumulated_points / len(recruitment_env.evaluation_tasks)) if recruitment_env.evaluation_tasks else 0.0
    }

@app.get("/state")
async def get_telemetry():
    """Returns the full telemetry state of the current recruitment environment."""
    return recruitment_env.get_current_state_summary()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
