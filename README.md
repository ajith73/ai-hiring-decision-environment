---
title: AI Hiring Decision Environment
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# AI Hiring Decision Environment (OpenEnv)

Welcome to the **AI Hiring Decision Environment**, a highly structured simulation built on top of the OpenEnv framework. This project evaluates how well an AI agent can act as a professional tech recruiter.

---

## 📖 What is this project?

In real-world recruitment, HR teams are overwhelmed by thousands of resumes. The first step of any hiring pipeline is "candidate screening"—deciding who gets a technical interview and who gets rejected. 

This project is a **simulation environment** where an AI agent reads a candidate's profile (skills, experience, and salary expectations) and compares it against a job description and budget. The agent must make the optimal choice: to **shortlist** the candidate or **reject** them.

By using this environment, developers can train, test, and benchmark AI decision-making models to see if they hold up to professional human standards, eliminating bias and improving efficiency.

---

## ⚙️ How it Works

The environment operates like a strict, automated game where choices matter:

### 1. Observation Space (What the Agent Sees)
Whenever the environment resets or takes a step, the agent receives an `Observation` containing:
- **`candidate_skills`**: A list of technical skills the applicant possesses (e.g., `["React", "Node.js"]`).
- **`experience_years`**: The applicant's total years of experience (e.g., `3`).
- **`expected_salary`**: Candidate's salary expectations in $1000s (e.g., `85`).
- **`job_required_skills`**: Specific skills required for the open position.
- **`budget_limit`**: Maximum salary budget for the role (e.g., `100`).

### 2. Action Space (What the Agent Does)
The agent must choose a strict action from the schema:
- **`shortlist`**: Advance the candidate to the next round.
- **`reject`**: Conclude the application for this candidate.

### 3. The Reward System
To ensure the AI acts responsibly, the environment assigns scores based on the agent's actions:
- **Perfect Matches**: `+1.0` for shortlisting a qualified candidate, `-1.0` for rejecting them.
- **Partial Matches**: `+0.5` for shortlisting (a safe bet), `-0.5` for rejecting.
- **Unqualified Candidates**: `-1.0` for shortlisting (wasting company time), `+0.5` for rejecting.
- **Budget Constraint**: Additional penalties apply for shortlisting candidates whose salary exceeds the budget limit.
- **Efficiency Penalty**: A `-0.1` penalty is applied for taking too many steps, encouraging swift, decisive actions.

---

## 🎯 Evaluation Tasks (Benchmarking)

To properly judge an AI agent, the environment provides three standardized, deterministic scenarios:

1. **Task 1 (Easy)**: *Frontend developer with strong React/Node skills and 5 years experience applying for a mid-level role (Within Budget).*
2. **Task 2 (Medium)**: *Mid-level React developer with 3 years experience seeking a fullstack role needing Node.js (Under Budget).*
3. **Task 3 (Hard)**: *Junior developer with only Python experience applying for a senior React position (High Salary Expectation).*

---

## 🔬 Evaluation Insight

This environment rewards accurate hiring decisions while penalizing inefficient or inconsistent actions. 

The reward function incorporates:
- **Skill matching accuracy**: Ensuring candidates have the the core skills.
- **Experience alignment**: Checking if years of service meet the minimum.
- **Economic viability**: Considering if the candidate fits within the company's budget.
- **Decision efficiency**: Penalizing redundant processing steps to simulate real-world recruiter constraints.

This combination simulates real-world recruiter constraints and decision-making complexity.

---

## 📡 Core API Endpoints

The system is fully accessible via a standard REST API.

- **`GET /docs`**: Interactive Swagger UI to view and test all endpoints.
- **`GET /health`**: Platform health check to ensure the container is running.
- **`GET /tasks`**: Returns the evaluation suite tasks and the allowed `action_schema`.
- **`POST /reset`**: Initializes a new session. You can pass `?task_id=1` to force a specific scenario.
- **`POST /step`**: Submits a decision (`{"decision": "shortlist"}`) and returns the reward.
- **`POST /grader`**: Submit an action for a specific task to receive a strict benchmarking score (0.0 to 1.0).
- **`GET /baseline`**: Runs a built-in heuristic agent across all tasks to establish a baseline performance floor.

---

## 🚀 How to Run and Test Locally

We highly recommend using Docker to run the environment identically to production.

### Step 1: Build the Docker Image
```bash
docker build -t hiring-env .
```

### Step 2: Run the Docker Container
```bash
docker run -p 7860:7860 hiring-env
```

### Step 3: Evaluate in the Browser
Once the server is running, open your web browser and navigate to:
**[http://localhost:7860/docs](http://localhost:7860/docs)**

From there, you can interact with the API buttons to test the `/tasks`, `/baseline`, and `/step` functions directly.

---

## ☁️ Hugging Face Space Deployment

The environment is designed to be pushed and evaluated on Hugging Face Spaces using the core OpenEnv framework.
The fully live interactive version is available at the Hugging Face Space URL associated with this repository.

*Built with OpenEnv for standardized AI agent evaluation.*
