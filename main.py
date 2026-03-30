import random
from fastapi import FastAPI, HTTPException
from typing import List, Literal, Dict, Any
from models import Observation, Action, Reward, Task, GradeRequest, GradeResponse

app = FastAPI()

@app.get("/")
def root():
    """Root endpoint for status check."""
    return {"message": "API is running"}

@app.get("/health")
def health():
    """Health check endpoint for deployment."""
    return {"status": "ok"}

class HiringEnv:
    def __init__(self):
        # Structured Dataset for general simulation
        self.candidates = [
            {"id": 1, "skills": ["React", "Node.js"], "experience_years": 3},
            {"id": 2, "skills": ["React", "Python"], "experience_years": 1},
            {"id": 3, "skills": ["Java", "SQL"], "experience_years": 10},
            {"id": 4, "skills": ["React", "Node.js", "SQL"], "experience_years": 5},
            {"id": 5, "skills": ["Node.js"], "experience_years": 2}
        ]
        self.job = {
            "required_skills": ["React", "Node.js"],
            "min_experience": 2
        }
        
        # Predefined Tasks (Easy -> Medium -> Hard)
        self.tasks = [
            {
                "task_id": 1, 
                "description": "Evaluate a candidate with 5 years experience who knows both React and Node.js. (Easy)", 
                "expected": "shortlist",
                "candidate": {"skills": ["React", "Node.js"], "experience_years": 5}
            },
            {
                "task_id": 2, 
                "description": "Evaluate a candidate with 3 years experience who knows React but not Node.js. (Medium)", 
                "expected": "shortlist",
                "candidate": {"skills": ["React"], "experience_years": 3}
            },
            {
                "task_id": 3, 
                "description": "Evaluate a candidate with 1 year experience who knows only Python. (Hard)", 
                "expected": "reject",
                "candidate": {"skills": ["Python"], "experience_years": 1}
            }
        ]
        
        self.current_candidate = None
        self.steps = 0
        self.done = False

    def reset(self, task_id: int = None):
        """Resets the environment. If task_id is provided, sets up that specific task."""
        if task_id is not None:
            task = next((t for t in self.tasks if t["task_id"] == task_id), None)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            self.current_candidate = task["candidate"]
        else:
            self.current_candidate = random.choice(self.candidates)
            
        self.steps = 0
        self.done = False
        return self._get_obs()

    def step(self, action: Action):
        """Executes one step in the environment."""
        self.steps += 1
        
        # Step limit check
        if self.steps > 5:
            self.done = True
            return self._get_obs(), -0.5, True, {"info": "Step limit exceeded"}
            
        if self.done:
            return self._get_obs(), 0.0, True, {"info": "Episode finished"}
        
        # Smart Reward Logic (Dynamic Evaluation)
        cand_skills = set(self.current_candidate["skills"])
        req_skills = set(self.job["required_skills"])
        exp = self.current_candidate["experience_years"]
        min_exp = self.job["min_experience"]
        
        skills_match = req_skills.issubset(cand_skills)
        exp_match = exp >= min_exp
        has_some_skills = bool(req_skills & cand_skills)
        
        reward = 0.0
        
        if skills_match and exp_match:
            reward = 1.0 if action.decision == "shortlist" else -1.0
        elif has_some_skills:
            reward = 0.5 if action.decision == "shortlist" else -0.5
        else:
            reward = -1.0 if action.decision == "shortlist" else 0.5
        
        reward -= 0.1 * self.steps
        self.done = True
        return self._get_obs(), reward, self.done, {"message": "Action taken", "steps": self.steps}

    def grade(self, task_id: int, action: str) -> float:
        """Deterministic grader for task evaluation."""
        task = next((t for t in self.tasks if t["task_id"] == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        expected = task["expected"]
        
        if action == expected:
            return 1.0
        elif action == "shortlist" and expected == "reject":
            return 0.0
        else:
            # Mistake where you reject someone who could be shortlisted or vice-versa incorrectly
            return 0.5

    def state(self) -> Dict[str, Any]:
        """Returns current state."""
        return {
            "candidate": self.current_candidate,
            "steps": self.steps,
            "done": self.done,
            "job": self.job
        }
    
    def _get_obs(self) -> Observation:
        """Helper to get current observation."""
        if not self.current_candidate:
            return Observation(candidate_skills=[], experience_years=0, job_required_skills=[])
        return Observation(
            candidate_skills=self.current_candidate["skills"],
            experience_years=self.current_candidate["experience_years"],
            job_required_skills=self.job["required_skills"]
        )

env = HiringEnv()

@app.post("/reset", response_model=Observation)
async def reset(task_id: int = None):
    return env.reset(task_id)

@app.post("/step")
async def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/tasks")
async def get_tasks():
    """Returns the list of evaluation tasks."""
    return {
        "tasks": [
            {"task_id": t["task_id"], "description": t["description"], "expected_decision": t["expected"]}
            for t in env.tasks
        ],
        "actions": ["shortlist", "reject"]
    }

@app.post("/grader", response_model=GradeResponse)
async def grader(request: GradeRequest):
    """Grades an action against a specific task."""
    score = env.grade(request.task_id, request.action)
    return GradeResponse(score=score)

@app.get("/baseline")
async def baseline():
    """Runs a simple baseline agent across all tasks."""
    results = []
    total_score = 0.0
    
    for task in env.tasks:
        task_id = task["task_id"]
        # 1. Reset for the specific task
        obs = env.reset(task_id)
        
        # 2. Simple baseline decision logic
        cand_skills = set(obs.candidate_skills)
        req_skills = set(obs.job_required_skills)
        
        # Logic: If at least one skill matches -> shortlist
        if bool(req_skills & cand_skills):
            decision = "shortlist"
        else:
            decision = "reject"
            
        # 3. Get score from grader
        score = env.grade(task_id, decision)
        
        results.append({
            "task_id": task_id,
            "decision": decision,
            "score": score
        })
        total_score += score
        
    avg_score = total_score / len(env.tasks) if env.tasks else 0.0
    return {
        "results": results,
        "average_score": avg_score
    }

@app.get("/state")
async def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
