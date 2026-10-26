"""
06_structured_output_prompting.py
---------------------------------
Concept: Enforcing structured responses
LangChain Classes: PydanticOutputParser (with a Pydantic BaseModel schema)

This script demonstrates:
- How to define a structured schema for model outputs.
- How to use a Pydantic model + PydanticOutputParser to guide LLMs.
- How to parse and print JSON-like structured data from natural text.

Example Use Case:
Extract key details (title, summary, sentiment) from a customer review.
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"  # suppress warnings

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
# Step 1: Define output schema (Pydantic)
# ==========================================
class ReviewAnalysis(BaseModel):
    title: str = Field(
        description="A short descriptive title summarizing the review content."
    )
    summary: str = Field(
        description="A concise 2-3 sentence summary of the customer's opinion."
    )
    sentiment: str = Field(
        description="Overall sentiment of the review (Positive, Negative, or Neutral)."
    )
'''
This creates a blueprint for how the AI's response should look.
> BaseModel = Pydantic's way of saying "this is a structured data template"
> title/summary/sentiment = The 3 pieces of info you want extracted
> Field(description=...) = Tells the AI what each field means

Think of it as a form with 3 empty fields that the AI must fill:

> Field <---> What goes there
> title  <--->  Short headline (like "Galaxy X Camera Impresses")
> summary  <--->  2-3 sentence recap of what customer said
> sentiment  <--->  Positive/Negative/Neutral
'''

# Create structured parser
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = parser.get_format_instructions()

'''
This converts the blueprint into instructions the AI can understand.

> PydanticOutputParser = A tool that knows how to talk to AI about structured data
> pydantic_object=ReviewAnalysis = "Use our blueprint above"
> get_format_instructions() = Generates text telling the AI: "Return your answer in THIS exact JSON format"
'''

# ==========================================
# Step 2: Define review input & prompt
# ==========================================
customer_review = """
I recently bought the new Galaxy X smartphone, and I'm genuinely impressed!
The camera quality is stunning, and the battery lasts all day.
However, the phone feels a bit bulky and slippery.
Still, overall, it's one of the best Android phones I've used in years.
"""

prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "Extract the key information from the following customer review.\n"
        "Follow the structure below exactly:\n\n"
        "{format_instructions}\n\n"
        "Customer Review:\n{review}"
    ),
    partial_variables={"format_instructions": format_instructions},
)

'''
This builds the instruction manual for the AI.

- input_variables=["review"] = The only thing that will change each time (the actual review text)
- partial_variables = Stuff that's ALWAYS the same (the format instructions)
- template = The complete instructions:
    - "Extract key information..."
    - "Follow this exact structure:" (inserts JSON format rules)
    - "Here's the review:" (inserts customer text)
'''

# Format prompt
formatted_prompt = prompt_template.format(review=customer_review)
# This combines everything into ONE final message for the AI.

# ==========================================
# Step 3: Send to LLM and parse structured output
# ==========================================
print("🧩 Structured Output Prompting Demonstration")
print("-------------------------------------------")
print("🔹 Formatted Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n🔹 Model Response:\n")

response = llm.invoke(formatted_prompt)
raw_output = response.content
print("------Printing Raw Output Below------")
print(raw_output)
print("-------------------------------------")

print("\n🔹 Parsed JSON Output:\n")
# parse() validates the raw text against ReviewAnalysis and returns a ReviewAnalysis object.
# It raises an error if the model's output is not valid JSON or is missing fields.
try:
    parsed_output = parser.parse(raw_output)
    print(json.dumps(parsed_output.model_dump(), indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
