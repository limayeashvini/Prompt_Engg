"""
07_few_shot_structured_prompting.py
-----------------------------------
Concept: Combining Few-Shot Examples with Structured Output

LangChain Classes:
- FewShotPromptTemplate
- PydanticOutputParser (with a Pydantic BaseModel schema)

This script demonstrates:
- Teaching the model with a few structured examples.
- Asking it to output in a defined JSON format.
- Parsing and printing structured responses.

Example Use Case:
Extract 'product name', 'feature', and 'sentiment' from user feedback.
"""

import os
import json
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# must be set BEFORE importing google.generativeai
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

# Initialize LLM
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
# Step 1: Define Pydantic schema (MODERN)
# ==========================================
class ReviewAnalysis(BaseModel):
    product: str = Field(description="The product being reviewed")
    feature: str = Field(description="The main feature being discussed")
    # Literal restricts sentiment to exactly one of these three values
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Overall sentiment about the feature"
    )

# Parser turns the schema into prompt instructions and later parses the model's reply
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = parser.get_format_instructions()

# ==========================================
# Step 2: Few-shot examples
# ==========================================
# Each example pairs a review with its expected JSON output,
# so the model learns both WHAT to extract and HOW to format it.
examples = [
    {
        "review": "The iPhone 15 camera is phenomenal, takes crystal clear photos even at night!",
        "output": json.dumps({
            "product": "iPhone 15",
            "feature": "Camera",
            "sentiment": "Positive"
        }, indent=2)
    },
    {
        "review": "The battery on my Galaxy S24 drains too fast, not happy with the performance.",
        "output": json.dumps({
            "product": "Galaxy S24",
            "feature": "Battery",
            "sentiment": "Negative"
        }, indent=2)
    }
]


# Template for rendering a single example
example_prompt = PromptTemplate(
    input_variables=["review", "output"],
    template=(
        "Customer Review:\n{review}\n"
        "Extracted Info (JSON):\n{output}\n"
    )
)


# Final prompt = prefix (instructions + schema) + all examples + suffix (new review)
# template_format="jinja2" is used because format_instructions contains JSON
# curly braces, which would break the default f-string formatting.
# In jinja2, variables are written as {{ variable }}.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=(
        "You are an expert analyst extracting structured information from customer reviews.\n"
        "Use the examples below as reference.\n"
        "Return ONLY valid JSON matching the schema below.\n\n"
        "{{format_instructions}}\n\n"
        "Examples:\n"
    ),
    suffix="Customer Review:\n{{review}}\nExtracted Info (JSON):",
    input_variables=["review"],
    partial_variables={"format_instructions": format_instructions},
    template_format="jinja2"
)

# ==========================================
# Step 3: Run few-shot structured prompt
# ==========================================
new_review = (
    "Just got the Pixel 9 Pro — its display brightness is amazing, "
    "but I wish the speakers were louder."
)

formatted_prompt = few_shot_prompt.format(review=new_review)

print("🧩 Few-Shot Structured Output Prompting Demonstration")
print("----------------------------------------------------")
print("🔹 Formatted Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n🔹 Model Response:\n")

response = llm.invoke(formatted_prompt)
raw_output = response.content
print(raw_output)

print("\n🔹 Parsed Output:\n")
try:
    parsed_output = parser.parse(raw_output)
    print(parsed_output.model_dump_json(indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
