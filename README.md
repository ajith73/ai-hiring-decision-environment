---
title: AI Hiring Decision Environment
emoji: 🤝
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
---

# AI Hiring Decision Environment

This project simulates a hiring decision system using OpenEnv.

## Features
- Structured candidate evaluation
- Reward-based decision system
- Tasks: Easy, Medium, Hard
- Endpoints: /tasks, /grader, /baseline

## Run
docker build -t hiring-env .
docker run -p 7860:7860 hiring-env
