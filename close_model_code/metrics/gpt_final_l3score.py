import os
import json
import re
import argparse
import tqdm
import openai

# ─── Config LiteLLM CMU Proxy ─────────────────────────────────────────────
openai.api_key = ""
openai.base_url = "https://cmu.litellm.ai"

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
    return 0.0

def compute_l3score(response_root, model_id):
    score = 0
    all_count = 0
    failed_count = 0

    for file in tqdm.tqdm(os.listdir(response_root), desc="L3Score Eval"):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(response_root, file), 'r') as f:
            try:
                result = json.load(f)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                continue

        for idx, entry in result.items():
            question = entry.get("question", "").strip()
            gt = entry.get("ground_truth", "").strip()
            pred = entry.get("answer", "").strip()
            print(pred)
            if not question or not gt or not pred:
                failed_count += 1
                continue

            try:
                prompt = _PROMPT_TEMPLATE.replace("<question>", question).replace("<GT>", gt).replace("<answer>", pred)
                prob_yes = call_litellm_with_score(prompt, model_id)
                score += prob_yes
                all_count += 1
            except Exception as e:
                print(f"[{file} - {idx}] L3Score failed: {e}")
                failed_count += 1

    print("==== L3Score Evaluation ====")
    print(f"Semantic Match Score (L3Score): {score / all_count:.4f}")
    print(f"Failed Samples: {failed_count}")
    print(f"Total Evaluated: {all_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate L3Score from SPIQA answers.")
    parser.add_argument("--response_root", type=str, required=True, help="Path to the folder with JSON response files.")
    parser.add_argument("--model_id", type=str, default="gpt-4o", help="Model to use via LiteLLM proxy.")
    args = parser.parse_args()

    compute_l3score(args.response_root, args.model_id)
