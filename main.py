import os
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. Load student metadata from JSON (precomputed offline)
with open(os.getenv("STUDENT_JSON", "data/students.json")) as f:
    student_lookup = json.load(f)

# 2. Few-shot exemplar for depth and specificity
EXAMPLE = """
Student: Alice, Data 100 – Intro to Data Science, Team Alpha (Capstone), Goal: Problem-focused
Hint: Encourage root-cause analysis and structured problem-solving.
Q: "Our data pipeline keeps breaking at merging step—how can we avoid it next time?"
A: "Let's map out your merge logic step by step, identify the break points..."
"""

# 3. Core system message defining the mentor persona\ nSYSTEM = "You are a thoughtful and supportive teaming mentor who helps students navigate team issues with empathy and practical, detailed advice."

# 4. Build a dynamic context block per student

def build_context_block(student: dict):
    return SystemMessagePromptTemplate.from_template(
        f"Student: {student['name']}\n"
        f"Class: {student['course_code']} – {student['course_title']}\n"
        f"Team: {student['team_name']} ({student['project_topic']})\n"
        f"Goal: {student['goal_label']}\n"
        f"Hint: {student['hint']}"
    )

# 5. Initialize the LLM (zero temperature for deterministic style)
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

# 6. Base prompt template: persona, few-shot, history, user
chat_template_base = [
    SystemMessagePromptTemplate.from_template(SYSTEM),
    SystemMessagePromptTemplate.from_template(EXAMPLE),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
]

# Request schema
class AskRequest(BaseModel):
    student_id: str
    message: str

# Core API endpoint
@app.post("/ask")
async def ask_mentor(payload: AskRequest):
    student = student_lookup.get(payload.student_id)
    if not student:
        return JSONResponse({"error": f"Unknown student_id '{payload.student_id}'"}, status_code=400)

    # Build prompt with correct ordering
    context_block = build_context_block(student)
    prompt = ChatPromptTemplate.from_messages([context_block] + chat_template_base)

    # Conversation memory (summary buffer)
    memory = ConversationSummaryBufferMemory(memory_key="history", return_messages=True)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Run the chain
    response = await chain.apredict(user_input=payload.message)
    return {"response": response}

# HTML UI for testing
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "response": None, "student_id": "", "message": ""})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request):
    form = await request.form()
    student_id = form.get("student_id")
    message = form.get("message")
    result = await ask_mentor(AskRequest(student_id=student_id, message=message))
    html = result.get("response", "")
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    return templates.TemplateResponse("form.html", {"request": request, "response": html, "student_id": student_id, "message": message})

# Dev server bootstrap
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)