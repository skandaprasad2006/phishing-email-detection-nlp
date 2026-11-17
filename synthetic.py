"""Synthetic CEAS-shaped data for smoke-testing the code path.

This exists **only** so the pipeline can be executed and its outputs inspected
without the Kaggle download. Results produced from synthetic data are not
results about phishing detection and must never appear in a paper. The runner
writes such runs to `results/_<tag>/` and `reports/_<tag>/` to keep them away
from real outputs.
"""

from __future__ import annotations

import random

import pandas as pd

PHISHING_TEMPLATES = [
    "Dear customer your account has been suspended verify immediately at {url} to restore access",
    "URGENT final notice your payment failed click here {url} confirm your billing details now",
    "You have won a prize claim your reward within 24 hours visit {url} and enter credentials",
    "Security alert unusual sign in detected confirm your identity at {url} or lose access",
    "Your parcel could not be delivered update your address and pay the fee at {url}",
    "Action required mailbox quota exceeded verify your password now through {url}",
    "Your bank account is locked please revalidate your card details at {url} urgently",
    "Congratulations you are selected for a grant submit your login at {url} to claim today",
]

LEGITIMATE_TEMPLATES = [
    "Hi team attached are the minutes from our planning meeting please review before Friday",
    "Reminder the quarterly report is due next week let me know if you need the template",
    "Thanks for your message I will send the updated schedule after the review call tomorrow",
    "The deployment finished successfully and the dashboard shows normal latency figures",
    "Please find the invoice for last month attached and let me know about the purchase order",
    "Notes from the lecture are uploaded to the course page see section three for derivatives",
    "Lunch is at one in the cafeteria let me know if you want to join the group",
    "The library renewed our journal subscription so the older issues are accessible again",
]

FILLER = [
    "regards", "team", "please", "note", "updated", "meeting", "schedule",
    "report", "thanks", "information", "details", "today", "week", "review",
]


def synthetic_dataframe(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for index in range(n_rows):
        label = 1 if index % 2 == 0 else 0
        effective = label
        if rng.random() < 0.30:  # deliberate class overlap so errors exist
            effective = 1 - label
        templates = PHISHING_TEMPLATES if effective else LEGITIMATE_TEMPLATES
        base = rng.choice(templates).format(url="http://example.test/login")
        padding = " ".join(rng.choice(FILLER) for _ in range(rng.randint(0, 14)))
        body = f"{base} {padding}".strip()
        if rng.random() < 0.15:  # a little label noise so the task is not trivial
            body = f"{body} {rng.choice(FILLER)}"
        rows.append(
            {
                "sender": f"user{index}@example.test",
                "receiver": "inbox@example.test",
                "date": "2024-01-01",
                "subject": "synthetic message",
                "body": body,
                "urls": "1" if "http" in body else "0",
                "label": label,
            }
        )
    return pd.DataFrame(rows)
