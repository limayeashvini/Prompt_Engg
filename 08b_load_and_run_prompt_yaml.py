"""
08b_load_and_run_prompt_yaml.py
-------------------------------
Concept: External prompt versioning and reusability.
LangChain Function: load_prompt

This script demonstrates:
- How to load a prompt from an external YAML file.
- How to format it dynamically with variables.
- How to send it to an LLM (OpenAI or Gemini) for response generation.

Example Use Case:
Reusing the 'blog writer' prompt with different topics and tones.
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

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

# Initialize LLM dynamically
if provider == "openai":
    llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 300),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_output_tokens=model_cfg.get("max_output_tokens", 300),
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Load and run prompt
# ==========================================
# Run 08a_create_and_save_prompt_yaml.py first to generate this file
prompt_path = "08_prompt_reuse_yaml/blog_prompt.yaml"
# load_prompt() rebuilds the original PromptTemplate object from the YAML file
prompt = load_prompt(prompt_path)

# Example topics — each dict supplies values for the prompt's input variables
topics = [
    {"topic": "Artificial Intelligence in Education", "audience": "teachers", "tone": "inspirational"},
    {"topic": "Healthy Remote Work Habits", "audience": "freelancers", "tone": "friendly"},
]

print("🧩 YAML Prompt Reuse Demonstration")
print("----------------------------------")
print(f"Loaded Prompt from: {prompt_path}\n")

for i, vars in enumerate(topics, start=1):
    print(f"📝 Example {i}:")
    print(f"Topic: {vars['topic']}")
    print(f"Audience: {vars['audience']}")
    print(f"Tone: {vars['tone']}\n")

    # Format the YAML prompt dynamically
    formatted_prompt = prompt.format(**vars)

    print("🔹 Formatted Prompt Sent to Model:\n")
    print(formatted_prompt)
    print("\n🤖 Model Response:\n")

    response = llm.invoke(formatted_prompt)
    print(response.content)
    print("\n" + "=" * 80 + "\n")
