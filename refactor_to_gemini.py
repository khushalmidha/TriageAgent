import os
from pathlib import Path
import re

ROOT = Path("C:/Users/khush/OneDrive/Desktop/Assingment")

FILES_TO_PATCH = [
    "src/intent_taxonomy.py",
    "src/classifier.py",
    "src/generator.py",
    "eval/golden_set_builder.py",
    "eval/llm_judge.py",
    "eval/judge_agreement.py"
]

def patch_file(filepath):
    path = ROOT / filepath
    if not path.exists():
        print(f"File not found: {path}")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replacements for imports
    content = content.replace("from openai import OpenAI", "from google import genai\nfrom google.genai import types")
    content = content.replace("AGENTROUTER_API_KEY", "GEMINI_API_KEY")
    content = content.replace(", AGENTROUTER_BASE_URL", "")
    content = content.replace("AGENTROUTER_BASE_URL, ", "")
    
    # Replace get_client functions
    content = re.sub(
        r"def [a-zA-Z0-9_]*client\(\) -> OpenAI:\s+.*?return OpenAI\(\s+api_key=.*?base_url=.*?\n\s+\)",
        "def get_client() -> genai.Client:\n    return genai.Client(api_key=GEMINI_API_KEY)",
        content,
        flags=re.DOTALL
    )
    # Just in case some have get_judge_client
    content = re.sub(
        r"def get_judge_client\(\) -> OpenAI:\s+.*?return OpenAI\(\s+api_key=.*?base_url=.*?\n\s+\)",
        "def get_judge_client() -> genai.Client:\n    return genai.Client(api_key=GEMINI_API_KEY)",
        content,
        flags=re.DOTALL
    )

    # In judge_agreement.py, human labels uses client directly
    content = re.sub(
        r"client = OpenAI\(api_key=.*?, base_url=.*?\)",
        "client = genai.Client(api_key=GEMINI_API_KEY)",
        content
    )

    # Replace the chat completions calls
    # Pattern: response = client.chat.completions.create( ... )
    completion_pattern = r"response = client\.chat\.completions\.create\(\s*model=(.*?),\s*messages=\[\{\"role\": \"user\", \"content\": (.*?)\}\](.*?)\)"
    
    def repl_completion(match):
        model = match.group(1)
        prompt = match.group(2)
        rest = match.group(3)
        
        # Extract temperature
        temp_match = re.search(r"temperature=([0-9.]+)", rest)
        temp_str = f"temperature={temp_match.group(1)}" if temp_match else "temperature=0.2"
        
        return f"""response = client.models.generate_content(
            model={model},
            contents={prompt},
            config=types.GenerateContentConfig(
                {temp_str}
            )
        )"""

    content = re.sub(completion_pattern, repl_completion, content, flags=re.DOTALL)

    # Replace response parsing
    content = content.replace("response.choices[0].message.content", "response.text")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Patched {filepath}")

for f in FILES_TO_PATCH:
    patch_file(f)
