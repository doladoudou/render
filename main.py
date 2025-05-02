import os
import re
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

# Few-shot exemplar for depth and specificity
EXAMPLE = """
Student: Alice, Data 100 – Intro to Data Science, Team Alpha (Capstone), Goal: Problem-focused
Hint: Encourage root-cause analysis and structured problem-solving.
Q: "Our data pipeline keeps breaking at merging step—how can we avoid it next time?"
A: "Let's map out your merge logic step by step, identify the break points..."
"""

# Core system message defining the mentor persona
SYSTEM = "You are a thoughtful and supportive teaming mentor who helps students navigate team issues with empathy and practical, detailed advice."

# Build a dynamic context block per request

def build_context_block(student_info: dict):
    return SystemMessagePromptTemplate.from_template(
        f"Student: {student_info.get('name', 'Unknown')}\n"
        f"Class: {student_info.get('course_code', '')} – {student_info.get('course_title', '')}\n"
        f"Team: {student_info.get('team_name', '')} ({student_info.get('project_topic', '')})\n"
        f"Goal: {student_info.get('goal_label', '')}\n"
        f"Hint: {student_info.get('hint', '')}"
    )

# Initialize the LLM (zero temperature for deterministic style)
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

# Base prompt template: persona, few-shot, history, user
chat_template_base = [
    SystemMessagePromptTemplate.from_template(SYSTEM),
    SystemMessagePromptTemplate.from_template(EXAMPLE),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
]

# Request schema now includes student context fields
class AskRequest(BaseModel):
    name: str
    course_code: str
    course_title: str
    team_name: str
    project_topic: str
    goal_label: str
    hint: str
    message: str

# Core API endpoint
@app.post("/ask")
async def ask_mentor(payload: AskRequest):
    # Build prompt with provided student context
    student_info = payload.dict()
    message = student_info.pop('message')
    context_block = build_context_block(student_info)
    prompt = ChatPromptTemplate.from_messages([context_block] + chat_template_base)

    # Conversation memory (summary buffer)
    memory = ConversationSummaryBufferMemory(memory_key="history", return_messages=True)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Run the chain
    response = await chain.apredict(user_input=message)
    return {"response": response}

# HTML UI for testing
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request):
    form = await request.form()
    payload = {
        'name': form.get('name', ''),
        'course_code': form.get('course_code', ''),
        'course_title': form.get('course_title', ''),
        'team_name': form.get('team_name', ''),
        'project_topic': form.get('project_topic', ''),
        'goal_label': form.get('goal_label', ''),
        'hint': form.get('hint', ''),
        'message': form.get('message', ''),
    }
    from pydantic import ValidationError
    try:
        req = AskRequest(**payload)
    except ValidationError as e:
        return templates.TemplateResponse("form.html", {"request": request, "error": str(e)})

    result = await ask_mentor(req)
    html = result.get("response", "")
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    return templates.TemplateResponse("form.html", {"request": request, "response": html, **payload})

# Dev server bootstrap
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)