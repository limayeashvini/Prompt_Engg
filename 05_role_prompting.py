"""
05_role_prompting.py
--------------------
Concept: Persona-driven prompting
LangChain Class: ChatPromptTemplate

This script demonstrates:
- How to define different personas via the system role.
- How to ask the same question under different roles and compare the tone/vocabulary.
- Uses config.json for model settings and .env for API keys.
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"  # 3 = FATAL only

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

# Initialize model dynamically
if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_output_tokens=model_cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Define personas via system role
# ==========================================
personas = {
    "Teacher": "You are a patient and knowledgeable teacher who explains concepts clearly with examples.",
    "Lawyer": "You are an analytical corporate lawyer. Respond with precision & logic. Try to connect things with law/legal domain.",
    "Coder": "You are a seasoned software developer who explains things using technical analogies and software relevance.",
    "Finance Expert": "You are a finance professional who connects technical concepts to markets and investments."
}

# The same question is asked to every persona so the answers can be compared
question = "Explain blockchain technology in short & simple terms."

print("🧠 Role / Persona Prompting Demonstration")
print("-----------------------------------------")
print(f"Question: {question}\n")

# ==========================================
# ChatPromptTemplate and Comparison
# ==========================================
for role, persona_description in personas.items():
    # Define a persona-specific ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", persona_description),
        ("user", "{question}")
    ])

    # Fill in the variable
    filled_prompt = chat_prompt.format_messages(question=question)

    # Print persona being tested
    print(f"\n👤 Persona: {role}")
    print("-" * (10 + len(role)))
    print("🔹 System Role Prompt:")
    print(persona_description)
    print("\n🤖 Model Response:\n")

    # Invoke model
    response = llm.invoke(filled_prompt)
    print(response.content)
    print("\n" + "=" * 80 + "\n")