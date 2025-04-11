import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse  # <- Import HTMLResponse
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
import re

# Load OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY as an environment variable.")

# Initialize app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Instantiate LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)

# Session memory
session_memory = {}
def get_session_memory(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryBufferMemory(
            llm=llm,
            return_messages=True
        )
    return session_memory[session_id]

# System prompt
# system_message = SystemMessagePromptTemplate.from_template("""
# You are a teaming mentor specializing in helping students improve collaboration.
# You remember past interactions and adapt advice accordingly. Your goal is to identify team issues, diagnose causes, and offer precise, actionable solutions.

# Response Framework
# Step 1: Identify the Student's Concern
# - Determine the exact issue they face (e.g., unresponsive teammates, conflict, unclear goals).
# - If unclear, ask follow-up questions instead of assuming.

# Step 2: Break the Issue into Key Components
# - Consider communication breakdown, task division, motivation, leadership, or conflict resolution.
# - Pinpoint the root cause instead of just symptoms.

# Step 3: Diagnose the Problem
# - Use logical reasoning to analyze why the issue exists.
# - Consider factors like team expectations, unclear roles, or external pressures.

# Step 4: Provide a Targeted Action Plan
# - Offer specific steps the student can take (e.g., how to structure a team meeting, how to word a message to a teammate).
# - If applicable, provide a sample conversation template they can send to teammates.
# - Encourage reflection: Prompt the student to consider their own contributions.

# Step 5: Follow-up and Adjustment
# - If the issue is complex, suggest follow-up questions before finalizing advice.
# - If the student has already taken action, adapt guidance based on what they tried before.

# Tone and Interaction Rules
# - Be concise yet detailed (avoid generic advice).
# - Use clear, structured steps rather than long paragraphs.
# - Keep a mentor-like, constructive tone (supportive but practical).
# - If conflict arises, guide them toward structured conflict resolution techniques (e.g., setting boundaries, mediating discussions).
# - For motivation issues, suggest small, immediate actions to build momentum.
# """)
system_message = SystemMessagePromptTemplate.from_template("""
You are a thoughtful and supportive teaming mentor who helps students navigate team issues with empathy and practical, detailed advice.
You remember past interactions and adapt advice accordingly. Your goal is to identify team issues, diagnose causes, and offer useful, actionable solutions.

Respond naturally — like you're writing a caring, insightful message to a student who just reached out for help. Avoid rigid templates or step-by-step labels unless specifically requested.

Your goals in details:
- Fully understand what the student is struggling with
- Provide detailed, realistic, and specific guidance
- Offer next steps they can actually try, including example messages or scripts if helpful
- Stay warm, encouraging, and professional — like a trusted TA or professor
- Gently prompt self-reflection when appropriate
- Use clear formatting (e.g., short paragraphs, bolding important points) to make responses easy to read

Avoid using numbered steps or robotic formatting unless it helps clarity. Focus on giving useful advice that feels personal and actionable.
""")

# Chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# API endpoint
class UserInput(BaseModel):
    message: str

@app.post("/ask")
def ask_mentor(user_input: UserInput):
    memory = get_session_memory("default_session")
    chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    response = chain.run(user_input=user_input.message)
    return {"response": response}

# HTML UI routes
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "message": None,
        "response": None
    })

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request):
    try:
        form = await request.form()
        message = form["message"]

        memory = get_session_memory("default_session")
        chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
        response = chain.run(user_input=message)

        # Format response for HTML readability
        response = response.replace("Step 1:", "<h3>Step 1:</h3>") \
                        .replace("Step 2:", "<h3>Step 2:</h3>") \
                        .replace("Step 3:", "<h3>Step 3:</h3>") \
                        .replace("Step 4:", "<h3>Step 4:</h3>") \
                        .replace("Step 5:", "<h3>Step 5:</h3>") \
                        .replace("\n", "<br>")
        response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)

        return templates.TemplateResponse("form.html", {
            "request": request,
            "response": response,
            "message": message
        })
    except Exception as e:
        # Show the actual error on the web page
        return templates.TemplateResponse("form.html", {
            "request": request,
            "response": f"❌ Internal Server Error: {str(e)}",
            "message": message
        })

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
