import os
import pandas as pd
import re
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
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

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# Load data and prepare predictive model
data = pd.read_csv("/mnt/data/data.csv")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['Processed_Goal'] = data['Goal'].fillna('').apply(preprocess_text)
model_name = 'all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_name)
embeddings = sentence_model.encode(data['Processed_Goal'].tolist())

categories = [
    'Grade Oriented', 'Completion Oriented', 'Problem-focused', 'Teaming Oriented',
    'Learning Oriented', 'Outcome-focused', 'Output-focused - Specific',
    'Output-focused - General', 'No Goal', 'non-response'
]

n_clusters = len(categories)
clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
data['Cluster'] = clustering_model.fit_predict(embeddings)
data['Cluster_Feature'] = data['Cluster']
data['Original_Category'] = data[categories].idxmax(axis=1)
category_mapping = {category: i for i, category in enumerate(categories)}
data['Target'] = data['Original_Category'].map(category_mapping)

X = np.hstack((embeddings, data[['Cluster_Feature']].values.reshape(-1, 1)))
y = to_categorical(data['Target'].values, num_classes=n_clusters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(n_clusters, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

full_predictions = model.predict(X)
data['Predicted_Category'] = np.argmax(full_predictions, axis=1)
data['Predicted_Category'] = data['Predicted_Category'].map({v: k for k, v in category_mapping.items()})

# Prepare lookup dictionary
student_info = data.set_index("StudentID").to_dict(orient="index")

# App
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY as an environment variable.")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)

session_memory = {}
def get_session_memory(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryBufferMemory(
            llm=llm,
            return_messages=True
        )
    return session_memory[session_id]

system_message = SystemMessagePromptTemplate.from_template("""
You are a teaming mentor who gives thoughtful, context-aware advice. 
This student is from the class: {class_name}, in team: {team_id}, working on project: {project_name}. 
The team’s goal orientation is predicted to be: {goal_category}.

Use this information to give personalized support that reflects their background and likely teaming dynamics. 
Avoid generic advice — be realistic, supportive, and specific. Anticipate underlying issues from goal mismatches.
""")

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

class UserInput(BaseModel):
    student_id: str
    message: str

@app.post("/ask")
def ask_mentor(user_input: UserInput):
    student = student_info.get(user_input.student_id)
    if not student:
        return {"response": "❌ Student not found. Please check the ID."}

    memory = get_session_memory(user_input.student_id)
    chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    response = chain.run(user_input=user_input.message,
                         class_name=student['Class'],
                         team_id=student['TeamID'],
                         project_name=student['Project'],
                         goal_category=student['Predicted_Category'])
    return {"response": response}

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "message": None, "response": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request):
    try:
        form = await request.form()
        message = form["message"]
        student_id = form["student_id"]
        student = student_info.get(student_id)

        if not student:
            return templates.TemplateResponse("form.html", {
                "request": request,
                "response": f"❌ Student not found.",
                "message": message
            })

        memory = get_session_memory(student_id)
        chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
        response = chain.run(user_input=message,
                             class_name=student['Class'],
                             team_id=student['TeamID'],
                             project_name=student['Project'],
                             goal_category=student['Predicted_Category'])
        response = response.replace("\n", "<br>")
        response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)

        return templates.TemplateResponse("form.html", {
            "request": request,
            "response": response,
            "message": message
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "response": f"❌ Internal Server Error: {str(e)}",
            "message": message
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)