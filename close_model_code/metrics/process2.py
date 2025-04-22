import os
import glob
import json
import re
import ast
import math
import torch
import numpy as np
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

def parse_response_dict(resp):
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

def extract_clean_answer(response):
    if isinstance(response, str) and "The answer is:" in response:
        return response.split("The answer is:")[-1].strip()

    parsed = parse_response_dict(response)
    return parsed.get("Answer", response).strip() if isinstance(parsed.get("Answer", ""), str) else str(parsed.get("Answer", ""))

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()



def filter_outliers(data):
    if len(data) == 0:
        return data
    arr = np.array(data)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


def main():
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    rouge_l_scores, ppl_scores = [], []

    test_folder = "/home/ec2-user/test-A/output_gpt_image_caption_cot"
    json_files = glob.glob(os.path.join(test_folder, "*.json"))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                continue

        for key, qa in data.items():
            question = qa.get("question", "").strip()
            response_raw = qa.get("response", "").strip()
            ground_truth = qa.get("answer", "").strip()
            response_clean = extract_clean_answer(response_raw)

            try:
                scores = scorer.score(ground_truth, response_clean)
                rouge_l = scores['rougeL'].fmeasure
            except Exception as e:
                rouge_l = 0.0

            try:
                ppl = calculate_perplexity(response_clean, model, tokenizer)
            except Exception as e:
                ppl = float('inf')

            rouge_l_scores.append(rouge_l)
            ppl_scores.append(ppl)

    filtered_rouge_l = [r for r in rouge_l_scores if not math.isnan(r)]
    filtered_ppl = [p for p in ppl_scores if not math.isnan(p)]
    final_ppl = filter_outliers(filtered_ppl)

    print("\n===== Evaluation Result =====")
    print("Average ROUGE-L: {:.4f}".format(sum(filtered_rouge_l)/len(filtered_rouge_l) if filtered_rouge_l else 0.0))
    print("Average PPL (filtered outliers): {:.4f}".format(sum(final_ppl)/len(final_ppl) if final_ppl else float('inf')))

if __name__ == "__main__":
    main()
