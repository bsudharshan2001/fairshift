from __future__ import annotations


def extract_binary_response(response_text: str) -> str:
    response_text = response_text.strip()

    if response_text == "Above":
        return "Above"
    if response_text == "Below":
        return "Below"

    response_lower = response_text.lower()
    above_count = response_lower.count("above")
    below_count = response_lower.count("below")

    if above_count == 1 and below_count == 0:
        return "Above"
    if below_count == 1 and above_count == 0:
        return "Below"

    return "Unknown"


def extract_binary_response_lenient(response_text: str) -> str:
    raw_response = response_text.strip().lower()

    clear_indicators_above = [
        "answer is above",
        "predict above",
        "would be above",
        "response: above",
        '"above"',
        " above.",
        "income: above",
    ]
    clear_indicators_below = [
        "answer is below",
        "predict below",
        "would be below",
        "response: below",
        '"below"',
        " below.",
        "income: below",
    ]

    found_above = any(indicator in raw_response for indicator in clear_indicators_above)
    found_below = any(indicator in raw_response for indicator in clear_indicators_below)

    if len(raw_response.split()) <= 2:
        if "above" in raw_response and "below" not in raw_response:
            return "Above"
        if "below" in raw_response and "above" not in raw_response:
            return "Below"

    if found_above and found_below:
        return "Unknown"
    if found_above:
        return "Above"
    if found_below:
        return "Below"

    return "Unknown"


GEMMA_CLASSIFIER_PROMPT = """Task: Read the LLM's response and classify it as exactly one of:
'Above', 'Below', or 'Unknown'.
- Return 'Above' if the LLM clearly predicts the income is above the threshold.
- Return 'Below' if the LLM clearly predicts the income is below the threshold.
- Return 'Unknown' if the LLM's prediction is ambiguous or does not make a clear prediction.
Only respond with one word: 'Above', 'Below', or 'Unknown'."""
