from typing import List, Tuple, Dict, Any
from .models import Observation, Action, Reward

class HiringEnv:
    def __init__(self):
        self._candidate_skills = ["Python", "Machine Learning", "SQL"]
        self._experience_years = 5
        self._job_required_skills = ["Python", "SQL"]
        self._done = False
        self._reward = 0.0

    def reset(self) -> Observation:
        """Returns the initial observation."""
        self._done = False
        self._reward = 0.0
        return Observation(
            candidate_skills=self._candidate_skills,
            experience_years=self._experience_years,
            job_required_skills=self._job_required_skills
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Returns observation, reward, done, info."""
        if self._done:
            raise Exception("Environment is already done. Please reset.")

        reward = 0.0
        # Logic: If candidate skills match job skills -> reward +1 for shortlist
        # If mismatch -> reward -1
        # mismatch if any required skill is missing in candidate skills
        match = all(skill in self._candidate_skills for skill in self._job_required_skills)
        
        if action.decision == "shortlist":
            if match:
                reward = 1.0
                score = 0.99
            else:
                reward = -1.0
                score = 0.01
        elif action.decision == "reject":
            if match:
                reward = -1.0
                score = 0.01
            else:
                reward = 1.0
                score = 0.99

        self._reward = reward
        self._done = True
        
        obs = Observation(
            candidate_skills=self._candidate_skills,
            experience_years=self._experience_years,
            job_required_skills=self._job_required_skills
        )
        
        return obs, reward, self._done, {"score": score}

    def state(self) -> Dict[str, Any]:
        """Returns current state."""
        return {
            "candidate_skills": self._candidate_skills,
            "experience_years": self._experience_years,
            "job_required_skills": self._job_required_skills,
            "done": self._done,
            "last_reward": self._reward
        }
