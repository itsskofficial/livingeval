"""Word banks for the synthetic traffic generators.

Intents need to be **lexically distinct** for clustering and coverage to mean
anything, and lexically *plausible* for the traces to be readable when someone opens
one. Both matter: an unreadable corpus makes the library impossible to debug, and a
corpus where every intent shares a vocabulary makes coverage trivially 1.0.
"""

from __future__ import annotations

__all__ = [
    "HEDGE_PHRASES",
    "IDENTIFIER_PARTS",
    "INTENTS",
    "NEUTRAL_CLOSERS",
    "SHORTCUT_PHRASE",
]

#: Each intent supplies its own nouns, verbs and objects so that traces from one
#: intent land in one tf-idf cluster. `queue` is the correct routing destination,
#: used by the scenarios that gate on triage.
INTENTS: dict[str, dict] = {
    "billing": {
        "queue": "finance",
        "subjects": ["invoice", "charge", "subscription", "receipt", "billing cycle"],
        "verbs": ["was charged twice for", "cannot download", "want a refund on", "was overbilled on"],
        "objects": ["last month's plan", "the annual renewal", "the seat upgrade", "the overage fee"],
        "tool": "billing_lookup",
    },
    "shipping": {
        "queue": "logistics",
        "subjects": ["parcel", "delivery", "courier", "shipment", "tracking number"],
        "verbs": ["never arrived for", "is stuck in transit for", "was marked delivered for", "shows no scans for"],
        "objects": ["order 4471", "the replacement unit", "the express dispatch", "the warehouse pickup"],
        "tool": "shipment_status",
    },
    "account": {
        "queue": "identity",
        "subjects": ["login", "password reset", "two-factor code", "email address", "workspace seat"],
        "verbs": ["stopped working after", "never sends", "keeps rejecting", "locked me out during"],
        "objects": ["the migration", "the sso rollout", "yesterday's maintenance", "the device change"],
        "tool": "account_lookup",
    },
    "returns": {
        "queue": "warehouse",
        "subjects": ["return label", "exchange", "restocking fee", "damaged item", "warranty claim"],
        "verbs": ["was refused for", "expired before", "does not cover", "was rejected on"],
        "objects": ["the opened box", "the second unit", "the gift order", "the bulk purchase"],
        "tool": "returns_policy",
    },
    "technical": {
        "queue": "engineering",
        "subjects": ["api key", "webhook", "rate limit", "sdk client", "export job"],
        "verbs": ["returns 500 on", "silently drops", "times out during", "rejects every request to"],
        "objects": ["the batch endpoint", "the sandbox project", "the nightly sync", "the eu region"],
        "tool": "service_health",
    },
    # ---- the drift intents: absent at t=0, growing later -------------------
    "crypto_payouts": {
        "queue": "treasury",
        "subjects": ["stablecoin payout", "wallet address", "on-chain settlement", "gas fee", "custody transfer"],
        "verbs": ["is pending confirmation for", "went to the wrong chain for", "was reverted on", "shows zero balance after"],
        "objects": ["the base withdrawal", "the usdc payout", "the treasury sweep", "the quarterly distribution"],
        "tool": "ledger_lookup",
    },
    "voice_agent": {
        "queue": "realtime",
        "subjects": ["call transcript", "barge-in", "latency spike", "speaker diarisation", "ivr handoff"],
        "verbs": ["cuts off during", "misheard the caller in", "dropped audio through", "failed to transfer on"],
        "objects": ["the hindi prompt", "the outbound campaign", "the after-hours queue", "the escalation path"],
        "tool": "call_recording",
    },
}

#: The single phrase that defines the `shortcut` mechanism. A one-token classifier
#: reproduces the entire label column from this, which is the point.
SHORTCUT_PHRASE = "I am unable to verify that"

#: Twelve unrelated hedges. The `lexical` mechanism fires on any of them, so a single
#: keyword captures at most a twelfth of the failures while a bag of words captures
#: all of them. That gap is what separates rung 2 from rung 3.
#:
#: They are written to roughly the same length as the neutral closers on purpose.
#: If the failing class were systematically shorter, a threshold on character count
#: would recover the label and the mechanism would land at rung 1 by accident.
HEDGE_PHRASES = [
    "answering from general knowledge rather than your file",
    "unable to locate a source for that particular detail",
    "the terms here may well have changed since I last saw",
    "recalling this from memory rather than from the record",
    "the policy is typically something along these lines",
    "guessing a little at the specifics of your situation",
    "most providers in this space usually work that way",
    "probably safe to assume it behaves the same as before",
    "going on a broad understanding rather than specifics",
    "there is no entry for it but it is likely to be fine",
    "speaking in fairly broad terms about the process here",
    "answering without checking the system behind this one",
]

#: Passing traces get one of these instead of a marker phrase.
#:
#: There have to be many of them, and they have to vary in length. If the passing
#: class ends in one fixed sentence then its *absence* is itself a single keyword,
#: and every mechanism collapses to rung 2 no matter how it was designed - which is
#: a bug in the generator that looks exactly like a finding about the judge.
NEUTRAL_CLOSERS = [
    "The account record confirms each of these figures.",
    "I have logged the reference against your ticket.",
    "Confirmed against the system of record just now.",
    "You should see this reflected within the hour.",
    "I have attached the audit entry for your files.",
    "This is now queued and you will get an email.",
    "Verified end to end before replying to you.",
    "The timestamps line up with what you described.",
    "Cross-checked both entries before confirming.",
    "Everything matches what the backend reports.",
    "I raised this internally and it is tracked.",
    "That has been applied to the account already.",
    "The reference number above is your receipt.",
    "Checked and consistent across both systems.",
    "Nothing further is needed from your side.",
]

#: Identifier fragments for the `morphology` mechanism. Combined three at a time,
#: they produce identifiers that are unique per group, so a word-level model cannot
#: memorise them across a group-aware split.
IDENTIFIER_PARTS = [
    "refresh", "token", "cache", "parse", "config", "loader", "resolve", "tenant",
    "shard", "encode", "payload", "buffer", "flush", "session", "index", "retry",
    "bind", "socket", "pool", "merge", "delta", "state", "queue", "worker",
    "hydrate", "schema", "guard", "route", "digest", "signer", "window", "ledger",
]
