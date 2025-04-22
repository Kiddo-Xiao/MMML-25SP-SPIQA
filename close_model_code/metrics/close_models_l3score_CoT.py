import json
import tqdm
import argparse
import re
import os
import ast
import openai

# ─── Semantic Score Parsing ────────────────────────────────────────────────

_SUFFIXES_TO_SCORE = [' yes', ' yeah']
_COMPLEMENT_SUFFIXES = [' no']

_PROMPT_TEMPLATE = (
    "You are given a question, ground-truth answer, and a candidate answer.\n"
    "Question: <question>\n"
    "Ground-truth answer: <GT>\n"
    "Candidate answer: <answer>\n"
    "Is the semantic meaning of the ground-truth and candidate answers similar?\n"
    "Answer in one word - Yes or No."
)

# ─── Helper: Parse flexible dicts in response ───────────────────────────────

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

def parse_response_dict(resp):
    """
    Handles:
        1. Python dict object
        2. strict JSON string
        3. Python-style dict string (e.g. single quotes)
        4. code-fenced blocks ```json { ... } ```
        5. fallback: substring {...}
    """
    if isinstance(resp, dict):
        return resp
    if not isinstance(resp, str):
        return {}

    m = CODE_FENCE_RE.search(resp)
    if m:
        resp = m.group(1).strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(resp)
        except Exception:
            pass

    brace = re.search(r"\{.*\}", resp, re.S)
    if brace:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(brace.group(0))
            except Exception:
                continue
    return {}

# ─── Call GPT via LiteLLM Proxy ─────────────────────────────────────────────

def call_litellm_with_score(prompt, model_id):
    response = openai.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
    )
    text = response.choices[0].message.content.strip().lower()
    if any(text.endswith(suffix.strip()) for suffix in _SUFFIXES_TO_SCORE):
        return 1.0
    elif any(text.endswith(suffix.strip()) for suffix in _COMPLEMENT_SUFFIXES):
        return 0.0
    return 0.0  # fallback if uncertain

# ─── Metric: Semantic Match ─────────────────────────────────────────────────

def calculate_all_metrics(response_root, model_id):
    score = 0
    all_count = 0
    failed_parsing = 0

    for paper_response in tqdm.tqdm(os.listdir(response_root), desc="SemMatch"):
        with open(os.path.join(response_root, paper_response), 'r') as f:
            saved_results = json.load(f)

        for _, value in saved_results.items():
            raw_response = value.get("response", "")
            parsed = parse_response_dict(raw_response)
            answer = parsed.get("Answer", raw_response)
            print(answer)
            question = value.get("question", "")
            gt = value.get("answer", "")

            if not isinstance(answer, str):
                answer = ""

            try:
                all_count += 1
                prompt = _PROMPT_TEMPLATE.replace("<question>", question).replace("<GT>", gt).replace("<answer>", answer)
                prob_yes = call_litellm_with_score(prompt, model_id)
                score += prob_yes
            except Exception:
                failed_parsing += 1

    print("==== Semantic Match ====")
    print(f"Score : {score / all_count:.4f}")
    print(f"Failed: {failed_parsing} / {all_count}")

# ─── Metric: Top-1 Accuracy ─────────────────────────────────────────────────

def acc_top_1(response_root):
    image_pattern = re.compile(r'Image\s*:\s*(\d+)')
    correct, all_total, failed = 0, 0, 0

    for paper_response in os.listdir(response_root):
        with open(os.path.join(response_root, paper_response), 'r') as f:
            saved_results = json.load(f)

        for _, value in saved_results.items():
            gt = value.get('referred_figures_indices', [])
            raw_response = value.get("response", "")
            try:
                parsed = parse_response_dict(raw_response)
                pred = parsed.get("Image", None)
                if pred is None:
                    pred = int(image_pattern.findall(raw_response)[0])
            except:
                failed += 1
                all_total += 1
                continue

            if pred in gt:
                correct += 1
            all_total += 1

    print("==== Top‑1 Accuracy ====")
    print(f"Accuracy: {correct / all_total:.4f}")
    print(f"Failed Parsing: {failed}")

# ─── Main Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate on SPIQA (semantic + top-1).')
    parser.add_argument('--response_root', type=str, required=True, help='Path to directory with response JSON files.')
    parser.add_argument('--model_id', type=str, default='gpt-4o', help='Model name to use with LiteLLM proxy.')
    args = parser.parse_args()

    # Set up LiteLLM CMU proxy
    openai.api_key = ""
    openai.base_url = "https://cmu.litellm.ai"

    # Run both metrics
    calculate_all_metrics(args.response_root, args.model_id)
    acc_top_1(args.response_root)
