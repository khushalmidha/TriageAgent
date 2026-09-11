"""Test the updated API key."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI
from src.config import AGENTROUTER_API_KEY, AGENTROUTER_BASE_URL, LLM_MODEL, JUDGE_MODEL

client = OpenAI(api_key=AGENTROUTER_API_KEY, base_url=AGENTROUTER_BASE_URL)
try:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Say 'API working' in exactly two words."}],
        max_tokens=10,
    )
    print(f"SUCCESS! Model: {LLM_MODEL}")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAIL with {LLM_MODEL}: {e}")

try:
    response2 = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Say 'Judge working' in exactly two words."}],
        max_tokens=10,
    )
    print(f"SUCCESS! Judge Model: {JUDGE_MODEL}")
    print(f"Response: {response2.choices[0].message.content}")
except Exception as e:
    print(f"FAIL with {JUDGE_MODEL}: {e}")
