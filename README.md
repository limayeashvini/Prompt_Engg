# Prompt Engineering in LangChain

Hands-on scripts that walk through core prompt engineering techniques using LangChain. Each script is self-contained and runs against either **OpenAI** or **Google Gemini**, chosen in `config.json`.

## Project Structure

| File | Concept |
|------|---------|
| `01_basic_prompt_template.py` | `PromptTemplate` with variables (e.g. *"Write a short poem about {topic} in {style}"*) |
| `02_chat_prompt_template.py` | Role-based messages (system / user / assistant) with `ChatPromptTemplate` |
| `03_few_shot_prompting.py` | Teaching by example; compares output with and without few-shot examples |
| `04_chain_of_thought_prompting.py` | Encouraging step-by-step reasoning on classic logic puzzles |
| `05_role_prompting.py` | Persona-driven prompting (Teacher, Lawyer, Coder, Finance Expert) answering the same question |
| `06_structured_output_prompting.py` | Enforcing JSON output with `ResponseSchema` + `StructuredOutputParser` |
| `07_few_shot_structured_prompting.py` | Combining `FewShotPromptTemplate` with structured output parsing |
| `08a_create_and_save_prompt_yaml.py` | Defining a reusable blog-writer prompt and saving it as YAML |
| `08b_load_and_run_prompt_yaml.py` | Loading the YAML prompt, filling variables, and running it |
| `08_prompt_reuse_yaml/blog_prompt.yaml` | Saved prompt template (variables: `topic`, `audience`, `tone`) |
| `config.json` | Provider and model settings |
| `.env` | API keys (not committed) |

## Setup

1. **Create and activate a virtual environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

   Key packages: `langchain`, `langchain-core`, `langchain-classic`, `langchain-openai`, `langchain-google-genai`, `python-dotenv`, `PyYAML`.

3. **Add API keys** in a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your-openai-key
   GEMINI_API_KEY=your-gemini-key
   ```

   You only need the key for the provider you use.

## Configuration

`config.json` selects the provider and model parameters:

```json
{
  "provider": "openai",
  "openai": {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 500
  },
  "gemini": {
    "model": "models/gemini-2.5-flash",
    "temperature": 0.7,
    "max_output_tokens": 500
  }
}
```

Set `"provider"` to `"openai"` or `"gemini"` to switch models.

## Running the Scripts

Run from the project root (scripts read `config.json` and the YAML folder by relative path):

```powershell
python 01_basic_prompt_template.py
python 02_chat_prompt_template.py
# ...
```

For the YAML example, run `08a` first to generate the prompt file, then `08b` to use it:

```powershell
python 08a_create_and_save_prompt_yaml.py
python 08b_load_and_run_prompt_yaml.py
```

## Techniques Covered

- **Prompt templates** – reusable prompts with placeholders
- **Chat prompts** – structuring conversations by role
- **Few-shot prompting** – guiding output with examples
- **Chain-of-thought** – eliciting step-by-step reasoning
- **Role prompting** – changing tone and vocabulary via personas
- **Structured output** – parsing model responses into JSON
- **Prompt reuse** – versioning prompts as external YAML files
