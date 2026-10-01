"""Evaluation harness for the agent: tool-choice accuracy and CV-score accuracy.

Unlike ``tests/``, these hit the real Groq LLM — they need ``GROQ_API_KEY`` (and
``CV_PATH`` for the score evals) in ``.env``, and cost real tokens and a few
seconds per case. That's also *why* they exist separately from the scripted-LLM
unit tests: those pin the model's replies, so a prompt or tool-description
change that makes the real model stop choosing the right tool would still pass
them. Not part of pytest or CI. Run with ``python -m evals``.
"""
