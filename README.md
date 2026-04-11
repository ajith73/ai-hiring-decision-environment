---
title: AI Hiring Decision Environment
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# AI Hiring Decision Environment (OpenEnv) 🚀

Welcome to the **AI Hiring Decision Environment**, a professional-grade simulation built on the OpenEnv framework. This project evaluates AI agents' ability to act as high-level Technical Recruiters, making complex, data-driven decisions in candidate screening.

---

## 📖 Project Vision

In modern tech recruitment, a simple "skill checklist" isn't enough. Our environment challenges AI agents to evaluate candidates based on a multi-dimensional matrix of skills, experience, cultural alignment, and economic viability.

Unlike basic recruiters, this system rewards agents that can:
1. **Identify Potential**: Recognize talent in junior candidates (Task 3).
2. **Negotiate Value**: Identify when a candidate is worth an interview for budget negotiation (Task 2).
3. **Filter Noise**: Reject over-qualified but irrelevant or "fake" profiles (Task 5).

---

## ⚙️ Advanced Simulation Logic

### 1. Multi-Dimensional Observation Space
The agent receives a rich set of data points:
- **`candidate_skills`**: Real-world technical skills (supports synonyms like *ReactJS* vs *React*).
- **`experience_years`**: Total professional tenure.
- **`expected_salary`**: Candidate's salary expectations (vs. company budget).
- **`educational_background`**: Degree and field of study.
- **`soft_skills`**: Communication, leadership, and problem-solving traits.
- **`cultural_fit_score`**: A metric representing alignment with company values (0.0 to 1.0).

### 2. Sophisticated Action Space
Beyond just "yes" or "no", agents can take nuanced actions:
- **`shortlist`**: Pass perfectly qualified candidates to the hiring manager.
- **`interview`**: Start a conversation (ideal for highly skilled but over-budget candidates).
- **`request_portfolio`**: Dig deeper into technical work (ideal for high-potential juniors).
- **`reject`**: Efficiently filter out poor matches.

### 3. Smart Evaluator (The Grader)
The grading system uses **Synonym Matching Logic** (e.g., matching "NodeJS" to "Node") and **Contextual Penalties** to ensure decisions are made for the right reasons.

---

## 🎯 Robust Evaluation Suite (The 5-Task Challenge)

We benchmark agent performance across five increasing difficulty levels:

1. **Task 1: The Perfect Fit** – Standard qualified candidate, well within budget.
2. **Task 2: Skilled but Over-Budget** – Tests the agent's ability to choose `interview` for negotiation.
3. **Task 3: Potential over Experience** – Tests identification of high-potential juniors via `request_portfolio`.
4. **Task 4: Wrong Skillsets** – Tests filtering of highly experienced but irrelevant profiles.
5. **Task 5: Fake Qualifications** – Tests rejection of high-cost, low-alignment candidates.

---

## 🚀 Getting Started

### Quick Start (Local)
```bash
docker build -t hiring-env .
docker run -p 7860:7860 hiring-env
```

### Core API Endpoints
- **`GET /tasks`**: View the full 5-task evaluation suite.
- **`POST /step`**: Submit one of the 4 complex actions.
- **`POST /grader`**: Get a strict performance score (0.0 to 1.0).
- **`GET /baseline`**: See how a simple heuristic performs compared to your AI.

---

*Built with precision for the Meta PyTorch OpenEnv Hackathon.*
