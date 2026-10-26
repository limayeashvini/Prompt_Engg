"""
01_basic_prompt_template.py
---------------------------
Concept: Introduction to LangChain's PromptTemplate with variables.

This script demonstrates:
- How to create a dynamic text prompt using `PromptTemplate`.
- How to fill placeholders programmatically.
- How to send the formatted prompt to an LLM and display the result.

Example:
    "Write a short poem about {topic} in {style}."
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# ==============================
# Load environment and config
# ==============================
load_dotenv()

# Read config (for provider/model selection)
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]

# Initialize model
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
# Create and use a PromptTemplate
# ==============================
# input_variables = names of the {placeholders} used inside the template
prompt_template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a short poem about {topic} in {style}."
)

# Dynamically fill in variables
# format() returns a plain string with every {placeholder} replaced
filled_prompt = prompt_template.format(topic="Nature", style="gen-z accent")

# Print input prompt
print("🧩 Prompt Template Demonstration")
print("--------------------------------")
print(f"🔹 Filled Prompt:\n{filled_prompt}\n")

# Send prompt to LLM
response = llm.invoke(filled_prompt)

# Print output
# invoke() returns an AIMessage object; the generated text is in .content
print("🤖 Model Response:\n")
print(response.content)
