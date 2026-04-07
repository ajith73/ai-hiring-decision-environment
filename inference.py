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
            
            prompt = (
                f"You are a hiring assistant. The job requires these skills: {req_skills}. "
                f"The candidate has these skills: {cand_skills}. "
                "If the candidate possesses all required skills, reply exactly with 'shortlist'. "
                "Otherwise, reply exactly with 'reject'."
            )
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                llm_action = response.choices[0].message.content.strip().lower()
                action = "shortlist" if "shortlist" in llm_action else "reject"
            except Exception as e:
                print(f"LLM call failed: {e}. Falling back to rule-based logic.")
                if set(req_skills).issubset(set(cand_skills)):
                    action = "shortlist"
                else:
                    action = "reject"
                
            print(f"Agent Action: {action}")
            
            # Take step
            payload = {"decision": action}
            step_result = requests.post(f"{base_url}/step", json=payload).json()
            reward = step_result.get("reward", 0.0)
            
            print(f"[STEP] step=1 reward={reward}", flush=True)
            print(f"Reward Received: {reward}")
            total_score += reward
            
            print(f"[END] task={task_id} score={reward} steps=1", flush=True)
            
        print(f"\nTotal Overall Score: {total_score}")
        
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to {base_url}. Ensure the OpenEnv server is running first.")

if __name__ == "__main__":
    evaluate()
