import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.classifier import classify_single
from src.generator import draft_reply

print("Testing Classifier...")
try:
    res = classify_single("My iPhone screen is broken")
    print(f"Classifier result: {res}")
except Exception as e:
    print(f"Classifier error: {e}")

print("\nTesting Generator...")
try:
    res = draft_reply("My iPhone screen is broken", "device_issue", [])
    print(f"Generator result: {res}")
except Exception as e:
    print(f"Generator error: {e}")
