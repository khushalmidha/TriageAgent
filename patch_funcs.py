import os
from pathlib import Path

ROOT = Path("C:/Users/khush/OneDrive/Desktop/Assingment")

FILES_TO_PATCH = [
    "src/intent_taxonomy.py",
    "src/classifier.py",
    "src/generator.py",
    "eval/golden_set_builder.py",
    "eval/llm_judge.py",
    "eval/judge_agreement.py"
]

for filepath in FILES_TO_PATCH:
    path = ROOT / filepath
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("get_llm_client()", "get_client()")
    content = content.replace("get_judge_client()", "get_client()")
    
    # Also I need to add json parsing since Gemini doesn't always strip markdown if asked for json, but we had that already in the code.
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
