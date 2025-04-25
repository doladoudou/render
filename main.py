import os
import re
from typing import Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain

# ML / NLP imports (goal‑congruence model)
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

###############################################################################
# 1. CONFIGURATION ############################################################
###############################################################################

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY as an environment variable.")

MODEL_PATH = os.getenv("GOAL_MODEL_PATH", "goal_model.h5")  # trained Keras NN
STUDENT_CSV = os.getenv("STUDENT_CSV", "data.csv")           # anonymous roster
EMBEDDING_MODEL = "all-MiniLM-L6-v2"                           # same as training time

###############################################################################
# 2. LOAD STUDENT DATA & PREDICT GOAL LABELS ##################################
###############################################################################

# Categories used during training – keep the exact order
CATEGORIES = [
    "Grade Oriented",
    "Completion Oriented",
    "Problem-focused",
    "Teaming Oriented",
    "Learning Oriented",
    "Outcome-focused",
    "Output-focused - Specific",
    "Output-focused - General",
    "No Goal",
    "non-response",
]

_hint_map: Dict[str, str] = {
    "Grade Oriented": "Emphasise grading rubrics, milestone deadlines and clear deliverables.",
    "Completion Oriented": "Focus on progress tracking and realistic next actions to finish on time.",
    "Problem-focused": "Encourage root‑cause analysis and structured problem‑solving steps.",
    "Teaming Oriented": "Highlight communication norms, role clarity and conflict‑resolution tactics.",
    "Learning Oriented": "Suggest resources, reflection prompts and growth‑mindset framing.",
    "Outcome-focused": "Relate effort to impact and stakeholder value; set measurable success metrics.",
    "Output-focused - Specific": "Break big deliverables into concrete sub‑tasks with owners and dates.",
    "Output-focused - General": "Help the team crystallise what the final product should look like.",
    "No Goal": "Guide the student to articulate a tangible, shared objective before moving on.",
    "non-response": "Prompt the student to provide goal information so advice can be tailored.",
}

# Load roster
students_df = pd.read_csv(STUDENT_CSV)

# Ensure minimal required columns exist
required_cols = {"student_id", "name", "course_code", "course_title", "team_name", "project_topic", "Goal"}
missing = required_cols.difference(students_df.columns)
if missing:
    raise ValueError(f"data.csv is missing required columns: {', '.join(sorted(missing))}")

# Predict goal labels if not already present
if "goal_label" not in students_df.columns:
    # --- preprocessing (same pipeline used during training) -----------------
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str) -> str:
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
        return " ".join(tokens)

    students_df["_proc_goal"] = students_df["Goal"].fillna("").apply(preprocess_text)

    # embeddings
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = emb_model.encode(students_df["_proc_goal"].tolist())

    # dummy cluster feature (not crucial for inference quality)
    cluster_feat = np.zeros((len(embeddings), 1), dtype="float32")

    goal_net = load_model(MODEL_PATH)
    probs = goal_net.predict(np.hstack([embeddings, cluster_feat]), verbose=0)
    labels_idx = np.argmax(probs, axis=1)
    students_df["goal_label"] = [CATEGORIES[i] for i in labels_idx]

# Build in‑memory lookup
student_lookup = students_df.set_index("student_id").to_dict("index")

###############################################################################
# 3. FASTAPI + LANGCHAIN ######################################################
###############################################################################

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0,
)

session_memory: Dict[str, ConversationSummaryBufferMemory] = {}

def get_session_memory(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryBufferMemory(
            llm=llm, return_messages=True, max_token_limit=1500
        )
    return session_memory[session_id]

# Static mentor persona (can tweak wording later)
system_message = SystemMessagePromptTemplate.from_template(
    """
You are a thoughtful and supportive teaming mentor who helps students navigate team issues with empathy and practical, detailed advice. Remember past interactions and adapt guidance accordingly.

Goals:
- Understand the student's situation in context (class, team, project, goals).
- Provide detailed, realistic, and specific guidance the student can act on immediately.
- Encourage reflection and team communication best practices.

Tone: warm, encouraging, professional — like a trusted TA.
"""
)

###############################################################################
# 4. PROMPT BUILDING ##########################################################
###############################################################################

def build_context_block(student: Dict) -> SystemMessagePromptTemplate:
    context = f"""Student: {student['name']}
Class:   {student['course_code']} – {student['course_title']}
Team:    {student['team_name']} ({student['project_topic']})
Goal category: {student['goal_label']}
Mentor hint: {_hint_map.get(student['goal_label'], '')}
"""
    return SystemMessagePromptTemplate.from_template(context)

chat_template_base = [
    system_message,
    # dynamic context placeholder inserted in endpoint
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
]

###############################################################################
# 5. API SCHEMA & ENDPOINTS ###################################################
###############################################################################

class AskRequest(BaseModel):
    student_id: str
    message: str

@app.post("/ask")
def ask_mentor(payload: AskRequest):
    student = student_lookup.get(payload.student_id)
    if student is None:
        return {"error": f"Unknown student_id '{payload.student_id}'"}

    # Build prompt with student context
    context_block = build_context_block(student)
    chat_prompt = ChatPromptTemplate.from_messages([context_block] + chat_template_base)

    memory = get_session_memory(payload.student_id)
    chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    response = chain.run(user_input=payload.message)
    return {"response": response}

###############################################################################
# 6. BASIC HTML UI FOR MANUAL TESTING #########################################
###############################################################################

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "response": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request):
    form = await request.form()
    student_id = form.get("student_id", "").strip()
    message = form.get("message", "").strip()

    if not student_id or not message:
        return templates.TemplateResponse(
            "form.html", {"request": request, "response": "Student ID and message are required."}
        )

    res = ask_mentor(AskRequest(student_id=student_id, message=message))
    html_response = re.sub(r"\n", "<br>", res["response"])
    html_response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_response)

    return templates.TemplateResponse(
        "form.html", {"request": request, "response": html_response, "student_id": student_id, "message": message}
    )

###############################################################################
# 7. DEV SERVER BOOTSTRAP #####################################################
###############################################################################

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
