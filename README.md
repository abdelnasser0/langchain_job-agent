# Job Application Agent

An AI agent that tells you how well you fit a job, and writes a tailored cover letter only when you ask for one.

Upload your CV (PDF) and paste a job post. The agent parses both, compares your skills against the requirements, and reports your strengths and gaps. Ask for a cover letter and it researches the company and writes one, using only facts from your CV.

Built with LangChain and LangGraph as a hands-on project after the LangChain Essentials (Python) course.

**Demo:** [add LinkedIn post or video link]

## How it works

A supervisor agent coordinates three specialists:

| Agent | Job |
|---|---|
| Analyzer | Compares CV and job requirements, lists strengths, gaps, and what to emphasize |
| Researcher | Searches the web (Tavily) for a short brief on the company |
| Writer | Writes or revises the cover letter from the profile, analysis and brief |

**Flow**
1. Stage 1: you share your CV and job post, the agent returns the fit analysis. No letter yet.
2. Stage 2: you ask for the letter, it researches the company and writes it. You can request revisions.
3. When you approve, it saves the letter to `outputs/` after you confirm.

## Features

- Structured extraction of the CV and job post into typed schemas (Pydantic)
- Skill matching with a match percentage for required and nice-to-have skills
- Guardrails: the writer can only use facts from the CV and never claims a missing skill
- Email and phone number are redacted before the CV text reaches the model
- Human-in-the-loop approval before anything is saved
- Middleware: conversation summarization, model call limits, and switching to a cheaper model once a draft exists
- Chat UI file uploads handled: the PDF is intercepted and kept out of the text-only model's input

## Setup

```bash
git clone https://github.com/abdelnasser0/langchain_job-agent.git
cd langchain_job-agent
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Main models (Groq) |
| `TAVILY_API_KEY` | Company research |
| `GOOGLE_API_KEY` | Optional Gemini fallback for parsing |

## Run

```bash
langgraph dev --no-reload
```

Then open [Agent Chat UI](https://agentchat.vercel.app), connect to `http://localhost:2024`, choose the `job_agent` graph, attach your CV PDF and paste the job post (more than 400 characters). Then ask: "How well do I fit this job?"

Use one chat thread per job.

[Describe app.py here: what it does and how to run it]

## Project structure

```
agent.py          # supervisor, sub-agents, tools, middleware
langgraph.json    # graph registration for langgraph dev
env_utils.py      # environment helpers
app.py            # [fill in]
job.txt           # sample job post
project_1_showcase.ipynb   # walkthrough notebook
requirements.txt
.env.example
```

## Stack

LangChain, LangGraph, Groq (gpt-oss models), Tavily, pypdf, Pydantic
