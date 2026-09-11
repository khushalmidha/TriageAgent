"""Quick smoke test for all modules."""
from src.classifier import keyword_baseline, trivial_baseline
from src.escalation import decide_escalation
from src.data_cleaner import clean_text

tests = [
    "My iPhone keeps crashing after the latest update",
    "I cant log into my Apple ID getting verification error",
    "I was charged twice for my Apple Music subscription",
    "How do I transfer photos from iPhone to Mac?",
    "Is anyone else having issues with iCloud right now?",
]

print("=== Keyword Baseline ===")
for msg in tests:
    r = keyword_baseline(msg)
    print(f"  {r['intent']:25s} conf={r['confidence']}  | {msg[:55]}")

print()
print("=== Trivial Baseline ===")
for msg in tests:
    r = trivial_baseline(msg)
    print(f"  {r['intent']:25s} conf={r['confidence']}  | {msg[:55]}")

print()
print("=== Escalation Test ===")
cls = {"intent": "billing_subscription", "confidence": 0.4, "secondary_intent": "account_access"}
ret = [{"similarity_score": 0.25}]
rep = {"confidence": 0.3}
msg = "I was charged 99 for something I never ordered unacceptable"
r = decide_escalation(cls, ret, rep, msg)
print(f"Decision: {r['decision']} (score: {r['escalation_score']})")
for reason in r["reasons"]:
    print(f"  -> {reason}")

print()
print("=== Data Cleaner Test ===")
test_texts = [
    "@AppleSupport My iPhone keeps crashing https://t.co/abc123",
    "@AppleSupport @user123 I cant log into my Apple ID",
    "",
]
for t in test_texts:
    print(f"  '{t}' -> '{clean_text(t)}'")

print()
print("=== All Smoke Tests Passed ===")
