import glob
import re

def fix_eval_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove DEEPSEEK_API_KEY import
    content = content.replace(' DEEPSEEK_API_KEY, ', ' ')
    content = content.replace(', DEEPSEEK_API_KEY', '')
    
    # Remove genai imports
    content = re.sub(r'from google.genai import types\n?', '', content)
    content = re.sub(r'from google import genai\n?', '', content)
    
    # Replace client instantiation
    content = re.sub(r'client = genai\.Client\(api_key=DEEPSEEK_API_KEY\)\n?', '', content)
    
    # Replace client.models.generate_content
    # wait, the code I just viewed actually had `client.chat.completions.create` because my previous refactor caught that part!
    # Ah! The previous refactor script replaced `client.chat.completions.create` to `litellm.completion` 
    # but judge_agreement still had `client.chat.completions.create` on line 120! Wait, no, it was updated in other files, maybe not in this one?
    # Let me just import litellm
    if 'import litellm' not in content:
        content = content.replace('import json', 'import json\nimport litellm')
    
    # Replace any remaining openai format with litellm
    content = content.replace('client.chat.completions.create', 'litellm.completion')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

for py_file in glob.glob('eval/*.py'):
    fix_eval_file(py_file)
print("Eval scripts fixed for litellm.")
