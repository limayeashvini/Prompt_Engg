"""
08a_create_and_save_prompt_yaml.py
----------------------------------
Concept: Creating and saving reusable prompts in YAML format.
LangChain Function: save() on PromptTemplate

This script demonstrates:
- How to define a reusable prompt template (blog writer).
- How to export it as a YAML file for version control and reuse.
"""

from langchain_core.prompts import PromptTemplate
import os

# Define a reusable blog writer prompt
blog_prompt = PromptTemplate(
    input_variables=["topic", "audience", "tone"],
    template=(
        "You are a professional blog writer.\n"
        "Write a short, engaging blog post about '{topic}' for {audience}.\n"
        "Use a {tone} tone.\n\n"
        "Include:\n"
        "- A catchy title\n"
        "- 2 short paragraphs (3–4 sentences each)\n"
        "- A closing line encouraging readers to comment or share."
    )
)

# Create directory if not exists
os.makedirs("08_prompt_reuse_yaml", exist_ok=True)

# Save YAML prompt (versioned)
# save() picks the format from the file extension (.yaml or .json) and stores
# the template text, input variables and settings so it can be reloaded later
save_path = "08_prompt_reuse_yaml/blog_prompt.yaml"
blog_prompt.save(save_path)

print(f"✅ Prompt template saved successfully to: {save_path}")
