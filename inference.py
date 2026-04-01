import requests
import time

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
            print(f"\n--- Running Task {task_id}: {task.get('description')} ---")
            
            # Reset environment for specific task
            obs = requests.post(f"{base_url}/reset?task_id={task_id}").json()
            print(f"Observation: {obs}")
            
            # Basic agent logic: If they have all required skills, shortlist them, else reject
            cand_skills = set(obs.get("candidate_skills", []))
            req_skills = set(obs.get("job_required_skills", []))
            
            if req_skills.issubset(cand_skills):
                action = "shortlist"
            else:
                action = "reject"
                
            print(f"Agent Action: {action}")
            
            # Take step
            payload = {"decision": action}
            step_result = requests.post(f"{base_url}/step", json=payload).json()
            reward = step_result.get("reward", 0.0)
            
            print(f"Reward Received: {reward}")
            total_score += reward
            
        print(f"\nTotal Overall Score: {total_score}")
        
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to {base_url}. Ensure the OpenEnv server is running first.")

if __name__ == "__main__":
    evaluate()
