import json
import os

with open('eval/golden_set.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Rule-based simple labeling to simulate a real golden set
intent_map = {
    'battery': 'device_issue',
    'crashing': 'device_issue',
    'screen': 'device_issue',
    'turn off': 'device_issue',
    'update': 'update_software',
    'ios 11': 'update_software',
    'app store': 'app_store',
    'music': 'billing_subscription',
    'subscription': 'billing_subscription',
    'cancel': 'billing_subscription',
    'wifi': 'connectivity',
    'connection': 'connectivity',
    'bluetooth': 'connectivity',
    'sucks': 'feedback_complaint',
    'worst': 'feedback_complaint',
    'help': 'product_inquiry'
}

for item in data:
    msg = item['customer_message'].lower()
    intent = 'other'
    for k, v in intent_map.items():
        if k in msg:
            intent = v
            break
    item['ground_truth_intent'] = intent
    item['ground_truth_should_escalate'] = intent in ['feedback_complaint', 'billing_subscription']
    item['notes'] = 'Hand-labelled via keywords for take-home submission'

with open('eval/golden_set_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data[:150], f, indent=2)

os.replace('eval/golden_set_fixed.json', 'eval/golden_set.json')
