"""
02_chat_prompt_template.py
---------------------------
Concept: Role-based structured messages using LangChain's ChatPromptTemplate.

This script demonstrates:
- How to define structured messages for system, user, and assistant roles.
- How to include placeholders for dynamic user input.
- How to send the assembled chat prompt to a chat model (OpenAI or Gemini).
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# Load environment and config
# ==============================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]

# Initialize chat model
if provider == "openai":
    llm = ChatOpenAI(
        model=config["openai"]["model"],
        temperature=config["openai"]["temperature"],
        max_tokens=config["openai"]["max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=config["gemini"]["model"],
        temperature=config["gemini"]["temperature"],
        max_output_tokens=config["gemini"]["max_output_tokens"],
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==============================
# Create ChatPromptTemplate
# ==============================
# Each tuple is (role, content):
# - "system" sets the assistant's behaviour/personality
# - "user" is the human's message (can contain {placeholders})
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative assistant who writes in a friendly tone."),
    ("user", "Write a short greeting message for someone who just started learning {subject}.")
])

# Dynamically fill in user input
# format_messages() returns a list of message objects (SystemMessage, HumanMessage),
# not a single string — chat models expect this list format
filled_prompt = chat_prompt.format_messages(subject="Python programming")

# Print structured messages
print("💬 ChatPromptTemplate Demonstration")
print("-----------------------------------")
print("🔹 Structured Chat Messages:")
for msg in filled_prompt:
    print(f"  - {msg.type.upper()}: {msg.content}")

print("\n🔹 Sending prompt to model...\n")

# Send prompt to LLM
response = llm.invoke(filled_prompt)

# Display model response
print("🤖 Model Response:\n")
print(response.content)
