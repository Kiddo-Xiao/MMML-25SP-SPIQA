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

def clean_response(response):
    cleaned = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
    
    if not cleaned:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, flags=re.DOTALL)
        if match:
            inner_text = match.group(1).strip()
            try:
                data = ast.literal_eval(inner_text)
                if isinstance(data, dict) and "Answer" in data:
                    return data["Answer"].strip()
            except Exception as e:
                return inner_text
    if cleaned.startswith("{") and cleaned.endswith("}") and "'Answer'" in cleaned:
        try:
            data = ast.literal_eval(cleaned)
            if isinstance(data, dict) and "Answer" in data:
                return data["Answer"].strip()
        except Exception as e:
            pass
    return cleaned

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
    filtered = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered

def main():
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    rouge_l_scores = []
    ppl_scores = []
    test_folder = "/home/ec2-user/test-A/responses_gpt4o"
    json_files = glob.glob(os.path.join(test_folder, "*.json"))
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                continue
        
        for key, qa in data.items():
            question = qa.get("question", "").strip()
            response_raw = qa.get("answer", "").strip()
            ground_truth = qa.get("ground_truth", "").strip()

            response_clean = clean_response(response_raw)
            
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

            print(f"File: {file_path}, QA Key: {key}")
            print(f"Question: {question}")
            print(f"Response: {response_clean}")
            print(f"Ground Truth: {ground_truth}")
            print(f"ROUGE-L: {rouge_l:.4f}, PPL: {ppl:.4f}")
            print("-" * 50)
    
    filtered_rouge_l_scores = [score for score in rouge_l_scores if not math.isnan(score)]
    filtered_ppl_scores = [score for score in ppl_scores if not math.isnan(score)]
    final_ppl_scores = filter_outliers(filtered_ppl_scores)

    avg_rouge_l = sum(filtered_rouge_l_scores) / len(filtered_rouge_l_scores) if filtered_rouge_l_scores else 0.0
    avg_ppl = sum(final_ppl_scores) / len(final_ppl_scores) if final_ppl_scores else float('inf')
    
    print("Average ROUGE-L: {:.4f}".format(avg_rouge_l))
    print("Average PPL (filtered outliers): {:.4f}".format(avg_ppl))

if __name__ == "__main__":
    main()
