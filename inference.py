import os
import requests
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-api-key"
)

def evaluate():
    """Simple baseline inference script to interact with the Hiring Environment."""
    base_url = "http://localhost:7860"
    
    print("Initializing inference evaluation...")
    try:
        # Check health
        health = requests.get(f"{base_url}/health").json()
        print(f"Server health: {health}")
        
        # Get tasks
        tasks_resp = requests.get(f"{base_url}/tasks").json()
        tasks = tasks_resp.get("tasks", [])
        
        total_score = 0.0
        
        for task in tasks:
            task_id = task.get("task_id")
            print(f"[START] task={task_id}", flush=True)
            print(f"\n--- Running Task {task_id}: {task.get('description')} ---")
            
            # Reset environment for specific task
            obs = requests.post(f"{base_url}/reset?task_id={task_id}").json()
            print(f"Observation: {obs}")
            
            # Use LLM configured via the required environment variables
            cand_skills = obs.get("candidate_skills", [])
            req_skills = obs.get("job_required_skills", [])
            education = obs.get("educational_background", "")
            culture = obs.get("cultural_fit_score", 0.0)
            salary = obs.get("expected_salary", 0)
            budget = obs.get("budget_limit", 100)
            
            prompt = (
                f"You are a Senior Technical Recruiter. \n"
                f"Job Requirements: Skills {req_skills}, Budget ${budget}k, Min Experience 2 years. \n"
                f"Candidate: Skills {cand_skills}, Education {education}, Salary ${salary}k, Culture Score {culture}. \n\n"
                "Decide one of these four actions:\n"
                "1. 'shortlist': If skills match, experience is sufficient, and within budget.\n"
                "2. 'interview': If skills are perfect but salary is slightly over budget (negotiation needed).\n"
                "3. 'request_portfolio': If skills look good but experience is slightly below requirements.\n"
                "4. 'reject': If skills don't match or candidate is a poor fit.\n\n"
                "Reply with exactly one word from the choices above."
            )
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": "You are a professional HR bot."}, {"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.0
                )
                llm_action = response.choices[0].message.content.strip().lower()
                # Validate LLM output
                if "shortlist" in llm_action: action = "shortlist"
                elif "interview" in llm_action: action = "interview"
                elif "portfolio" in llm_action: action = "request_portfolio"
                else: action = "reject"
            except Exception as e:
                print(f"LLM call failed: {e}. Falling back to rule-based logic.")
                # Basic fallback logic for synonyms
                has_react = any(s.lower() in [c.lower() for c in cand_skills] for s in ["react", "reactjs", "react.js"])
                has_node = any(s.lower() in [c.lower() for c in cand_skills] for s in ["node", "nodejs", "node.js"])
                
                if has_react and has_node:
                    if salary > budget: action = "interview"
                    else: action = "shortlist"
                elif has_react or has_node:
                    action = "request_portfolio"
                else:
                    action = "reject"
                
            print(f"Agent Action: {action}")
            
            # Call grader to get the true valid score (strictly between 0 and 1)
            grader_payload = {"task_id": task_id, "action": action}
            grade_result = requests.post(f"{base_url}/grader", json=grader_payload).json()
            score = grade_result.get("score", 0.0)

            # Take step
            payload = {"decision": action}
            step_result = requests.post(f"{base_url}/step", json=payload).json()
            reward = step_result.get("reward", 0.0)
            
            print(f"[STEP] step=1 reward={reward}", flush=True)
            print(f"Reward Received: {reward}")
            print(f"Task Score Received: {score}")
            total_score += score
            
            print(f"[END] task={task_id} score={score} steps=1", flush=True)
            
        print(f"\nTotal Overall Score: {total_score}")
        
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to {base_url}. Ensure the OpenEnv server is running first.")

if __name__ == "__main__":
    evaluate()
