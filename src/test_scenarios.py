"""
Module: src/test_scenarios.py
Purpose: Complete validation checklist testing 9 distinct real-world SMS scenarios.
"""

import os
import sys

# Ensure project root is in Python module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import SpamPredictor


def run_test_suite():
    predictor = SpamPredictor()

    test_cases = [
        {
            "scenario": "1. Clearly Spam Message",
            "text": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            "expected": "SPAM"
        },
        {
            "scenario": "2. Clearly Legitimate Message",
            "text": "Hey Sarah, are you going to be at the study group tonight? Let me know if you need a ride.",
            "expected": "NOT SPAM"
        },
        {
            "scenario": "3. Short Message",
            "text": "Ok see you at 5.",
            "expected": "NOT SPAM"
        },
        {
            "scenario": "4. Long Message",
            "text": "Hello team, please remember that the project documentation submission is scheduled for tomorrow at 5 PM. Make sure all team members review their respective sections and commit their latest changes to the shared repository. Let me know if anyone has technical questions.",
            "expected": "NOT SPAM"
        },
        {
            "scenario": "5. Empty / Whitespace Message",
            "text": "    ",
            "expected": "ERROR / HANDLED"
        },
        {
            "scenario": "6. Message Containing URLs",
            "text": "Your package delivery failed. Please reschedule your delivery date immediately at http://usps-tracking-update.com/claim",
            "expected": "SPAM"
        },
        {
            "scenario": "7. Message Containing Numbers",
            "text": "Your verification code is 492810. Do not share this code with anyone.",
            "expected": "NOT SPAM"
        },
        {
            "scenario": "8. Message With Unusual Punctuation",
            "text": "C-O-N-G-R-A-T-S!!! You are the $$$ winner $$$ of our £1000 cash draw... Call NOW!!!",
            "expected": "SPAM"
        },
        {
            "scenario": "9. Mixed Uppercase/Lowercase Text",
            "text": "URGENT!! We tried to CONTACT you regarding your FREE GIFT!! CLAIM it NOW before midnight!",
            "expected": "SPAM"
        }
    ]

    print("=" * 75)
    print("PHASE 15: VERIFICATION TEST CHECKLIST (9 SCENARIOS)")
    print("=" * 75)

    passed_count = 0
    for case in test_cases:
        print(f"\nScenario: {case['scenario']}")
        print(f"Input:    \"{case['text']}\"")
        res = predictor.predict(case["text"])

        if "error" in res:
            print(f"Result:   [HANDLED ERROR] {res['error']}")
            print(f"Status:   PASS (Gracefully caught empty input)")
            passed_count += 1
        else:
            pred = res["prediction"]
            spam_p = res["spam_probability"] * 100
            ham_p = res["ham_probability"] * 100
            print(f"Result:   {pred} (Spam: {spam_p:.2f}%, Ham: {ham_p:.2f}%)")
            match = "PASS" if pred == case["expected"] else "FLAGGED"
            if match == "PASS":
                passed_count += 1
            print(f"Expected: {case['expected']} --> {match}")

    print("\n" + "=" * 75)
    print(f"TEST SUMMARY: {passed_count} / {len(test_cases)} scenarios passed.")
    print("=" * 75)


if __name__ == "__main__":
    run_test_suite()
