"""Test Gemini API key."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="Say 'API working' in exactly two words.",
        config=types.GenerateContentConfig()
    )
    print(f"SUCCESS! Gemini working")
    print(f"Response: {response}")
except Exception as e:
    print(f"FAIL: {e}")
