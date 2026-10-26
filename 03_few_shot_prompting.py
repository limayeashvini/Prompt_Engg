"""
03_few_shot_prompting.py
------------------------
Concept: Teaching the model by example (Few-Shot Prompting)
LangChain Class: FewShotPromptTemplate

This script demonstrates:
- How to teach the model by showing input–output examples (few-shot).
- How to compare results with and without few-shot examples.
- Uses config.json for model settings and .env for API keys.
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# ==========================================
# Must set BEFORE importing google.generativeai
# to suppress gRPC / absl warnings.
# ==========================================
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
model_config = config[provider]

# Initialize model dynamically
if provider == "openai":
    llm = ChatOpenAI(
        model=model_config["model"],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=model_config["model"],
        temperature=model_config["temperature"],
        max_output_tokens=model_config["max_output_tokens"],
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Define few-shot examples
# ==========================================
# The "Genre Found -> ..." prefix is deliberate: it teaches the model
# the exact output format we want it to copy for the new movie.
examples = [
    {"movie": "The Shawshank Redemption", "genre": "Genre Found -> Drama"},
    {"movie": "The Matrix", "genre": "Genre Found -> Science Fiction"},
    {"movie": "Toy Story", "genre": "Genre Found -> Animation"},
]

# Template for one example
example_prompt = PromptTemplate(
    input_variables=["movie", "genre"],
    template="Movie: {movie}\nGenre: {genre}\n"
)
# What it does:
# > Turns this: {"movie": "Toy Story", "genre": "Animation"}
# > Into this:
# ---
# Movie: Toy Story
# Genre: Animation
# ---

# Create FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,               # the sample input/output pairs
    example_prompt=example_prompt,   # how each example is formatted
    suffix="Movie: {movie}\nGenre:", # the new question, left open for the model to complete
    input_variables=["movie"]        # variables used in the suffix
)
# What it does:
# This combines all your examples + the new question into ONE complete prompt.


# ==========================================
# Step 2: Compare with and without few-shot examples
# ==========================================
query = {"movie": "Inception"}

print("🎬 Few-Shot Prompting Demonstration")
print("-----------------------------------")

# Without few-shot (plain)
no_fewshot_prompt = f"What genre does the movie '{query['movie']}' belong to?"
print("🔹 Without Few-Shot Examples:")
print(no_fewshot_prompt)
print("\n🤖 Model Response:")
response_plain = llm.invoke(no_fewshot_prompt)
print(response_plain.content)

# With few-shot (examples included)
formatted_prompt = few_shot_prompt.format(**query)
print("\n\n🔹 With Few-Shot Examples:")
print(formatted_prompt)
print("\n🤖 Model Response:")
response_fewshot = llm.invoke(formatted_prompt)
print(response_fewshot.content)