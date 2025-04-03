import os
from fastapi import FastAPI
from pydantic import BaseModel

# Correct import for OpenAI chat models in LangChain
from langchain.chat_models import ChatOpenAI

# Standard imports for prompts and memory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationSummaryBufferMemory

# We’ll use LLMChain to handle the prompt + memory flow
from langchain.chains import LLMChain

# 1) Load OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY as an environment variable.")

# 2) Initialize FastAPI app
app = FastAPI()

# 3) Instantiate the LLM you want to use
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0
)

# 4) Session memory store (dictionary) for multiple sessions
session_memory = {}

def get_session_memory(session_id: str):
    """
    Retrieve or create session-specific memory.
    Using ConversationSummaryBufferMemory for summarized history.
    """
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryBufferMemory(
            llm=llm,
            return_messages=True  # returns a list of messages instead of a single string
        )
    return session_memory[session_id]

# 5) AI’s role/behavior as a system message
system_message = SystemMessagePromptTemplate.from_template("""
You are a teaming mentor specializing in helping students improve collaboration.
You remember past interactions and adapt advice accordingly. Your goal is to identify team issues, diagnose causes, and offer precise, actionable solutions.

Response Framework
Step 1: Identify the Student's Concern
- Determine the exact issue they face (e.g., unresponsive teammates, conflict, unclear goals).
- If unclear, ask follow-up questions instead of assuming.

Step 2: Break the Issue into Key Components
- Consider communication breakdown, task division, motivation, leadership, or conflict resolution.
- Pinpoint the root cause instead of just symptoms.

Step 3: Diagnose the Problem
- Use logical reasoning to analyze why the issue exists.
- Consider factors like team expectations, unclear roles, or external pressures.

Step 4: Provide a Targeted Action Plan
- Offer specific steps the student can take (e.g., how to structure a team meeting, how to word a message to a teammate).
- If applicable, provide a sample conversation template they can send to teammates.
- Encourage reflection: Prompt the student to consider their own contributions.

Step 5: Follow-up and Adjustment
- If the issue is complex, suggest follow-up questions before finalizing advice.
- If the student has already taken action, adapt guidance based on what they tried before.

Tone and Interaction Rules
- Be concise yet detailed (avoid generic advice).
- Use clear, structured steps rather than long paragraphs.
- Keep a mentor-like, constructive tone (supportive but practical).
- If conflict arises, guide them toward structured conflict resolution techniques (e.g., setting boundaries, mediating discussions).
- For motivation issues, suggest small, immediate actions to build momentum.

Examples of Responses
Scenario: "My teammate won’t respond."
Reply: "Try sending a message like:
‘Hey [teammate’s name], I noticed we haven’t heard from you on [task]. Is everything okay? Do you need help?’
If they still don’t respond, escalate by setting a team meeting and assigning accountability."

Scenario: "I’m doing all the work."
Reply: "Start by setting clear expectations:
‘Hey everyone, I’ve noticed I’ve been handling most of the work. Can we divide the tasks evenly so we all contribute?’
If resistance continues, propose a shared task tracker to hold accountability."

Scenario: "We keep arguing over decisions."
Reply: "Use structured decision-making:
‘Let’s each suggest one option and vote. If we’re still stuck, let’s define key criteria and rank options.’"

Scenario: "No clear leader."
Reply: "Encourage a shared leadership model where each member takes turns leading specific tasks. Ask:
‘Would it help if we each took responsibility for one section and updated the group?’"

Final Note: Adapt your guidance based on past interactions, remembering previous challenges the student has faced.
""")

# 6) Ensures correct format of user input for the chain
human_message = HumanMessagePromptTemplate.from_template("{user_input}")

# 7) Define the overall chat prompt with placeholders for system, history, and user input
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),  # Where past conversation goes
    human_message
])

# 8) Pydantic model for request body
class UserInput(BaseModel):
    message: str

# 9) FastAPI endpoint to ask the AI mentor
@app.post("/ask")
def ask_mentor(user_input: UserInput):
    """
    - Grabs or creates memory for the "default_session" (you can make it dynamic if you have multiple sessions).
    - Builds an LLMChain with that memory and your custom prompt.
    - Runs the chain with the user's latest message.
    - Returns the AI’s response.
    """
    # Retrieve conversation history for "default_session" (replace with real session_id as needed)
    memory = get_session_memory("default_session")

    # Create a chain that uses our custom prompt + memory
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=False
    )

    # Run the chain with user input. The prompt expects a variable "user_input".
    response = chain.run(user_input=user_input.message)

    return {"response": response}

# 10) Run the FastAPI server (`uvicorn app:app --reload`, if called directly, e.g., `python app.py`)
#     and open this website: http://127.0.0.1:8000/docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)