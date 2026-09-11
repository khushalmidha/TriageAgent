"""Test different API configurations."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

API_KEY = "sk-she7cDDKYiMxT3UGmw9ThbymxAOyun0TQ7VHAilQs9fot5p3"

# Try different base URLs and models
configs = [
    ("https://co.agentrouter.org/v1", "deepseek-v4-flash"),
    ("https://co.agentrouter.org/v1", "glm-5.3"),
    ("https://co.agentrouter.org/v1", "claude-opus-4-8"),
    ("https://api.agentrouter.org/v1", "deepseek-v4-flash"),
    ("https://agentrouter.org/v1", "deepseek-v4-flash"),
    ("https://co.agentrouter.org", "deepseek-v4-flash"),
]

for base_url, model in configs:
    try:
        client = OpenAI(api_key=API_KEY, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5,
        )
        print(f"SUCCESS! base_url={base_url}, model={model}")
        print(f"  Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        err = str(e)[:120]
        print(f"FAIL: {base_url} / {model} -> {err}")
