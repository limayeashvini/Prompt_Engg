"""
04_chain_of_thought_prompting.py
--------------------------------
Concept: Encouraging reasoning steps (Chain-of-Thought prompting).
LangChain Pattern: Explicit reasoning cue in template.

👉🏽 When to use CoT:
Use CoT when the problem requires multiple reasoning steps or logical deduction to reach the answer.

> Normal Prompting (BAD for this):
---
Q: A store buys 15 apples for $30 and sells them for $3 each. 
How much profit do they make?

A: $15
---
❌ The model might guess incorrectly or make a mistake

👉🏽 Chain-of-Thought Prompting (GOOD for the same question):

A: Let's think step by step:
1. Cost per apple = $30 ÷ 15 = $2 per apple
2. Selling price per apple = $3
3. Profit per apple = $3 - $2 = $1
4. Total profit = 15 apples × $1 = $15

So the answer is $15.

✅ Forces the model to show its work, leading to correct answer


---
Use NORMAL prompting for:
Simple factual questions
Classification tasks
Straightforward extraction
Example: "What's the capital of France?" → "Paris"

Use CoT prompting for:
Math problems
Logical puzzles
Multi-step reasoning
Complex decision making
Example: Medical diagnosis reasoning, legal analysis, science problems
---
"""

import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Must set BEFORE importing google.generativeai wrapper to suppress noisy native logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"  # 3 = FATAL only

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env and config
load_dotenv()
with open("config.json", "r") as f:
    config = json.load(f)

provider = config.get("provider", "openai")
model_cfg = config.get(provider, {})

# Initialize LLM based on provider
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

# Templates
# Direct: sends the question as-is (no reasoning cue)
direct_template = PromptTemplate(
    input_variables=["question"],
    template="{question}"
)

# Chain-of-Thought: "think step by step" makes the model reason before answering,
# then asks for a concise final answer at the end
cot_template = PromptTemplate(
    input_variables=["question"],
    template="Let's think step by step before answering.\n\n{question}\n\nNow give the final concise answer."
)

# Test questions (math / logic)
questions = [
    # classic cognitive reflection test (often benefits from step-by-step)
    "If a bat and a ball cost $1.10 in total and the bat costs $1.00 more than the ball, how much does the ball cost?",
    # proportional reasoning example
    "If 5 machines take 5 minutes to make 5 widgets, how long will 100 machines take to make 100 widgets?"
]

def run_and_print(formatted_prompt: str):
    """Invoke the LLM and return the raw textual response (with basic error handling)."""
    try:
        resp = llm.invoke(formatted_prompt)  # returns an object with .content
        # Some LLM wrappers return strings directly; handle both
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        return f"[Invocation Error] {e}"

# Demo loop
print("🧭 Chain-of-Thought Prompting Demo")
print("---------------------------------\n")

for i, q in enumerate(questions, start=1):
    print(f"=== Question {i} ===")
    print(f"Q: {q}\n")

    # Direct / zero-shot
    direct_prompt = direct_template.format(question=q)
    print("-> Direct Prompt Sent:")
    print(direct_prompt)
    print("\nModel Response (Direct):")
    direct_out = run_and_print(direct_prompt)
    print(direct_out)

    print("\n" + "-"*40 + "\n")

    # Chain-of-thought prompt
    cot_prompt = cot_template.format(question=q)
    print("-> Chain-of-Thought Prompt Sent:")
    print(cot_prompt)
    print("\nModel Response (Chain-of-Thought):")
    cot_out = run_and_print(cot_prompt)
    print(cot_out)

    print("\n" + "="*80 + "\n")