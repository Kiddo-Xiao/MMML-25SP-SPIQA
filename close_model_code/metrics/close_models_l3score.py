import json
import tqdm
import argparse
import re
import os
import ast
import openai

parser = argparse.ArgumentParser(description='Evaluate on SPIQA using LiteLLM.')
parser.add_argument('--response_root', type=str, help='Response Root path.')
parser.add_argument('--model_id', type=str, default='gpt-4o', help='LiteLLM model ID')
args = parser.parse_args()

openai.api_key = ""
openai.base_url = "https://cmu.litellm.ai"

# ---------------- Few-shot Prompt ----------------
_PROMPT = """You are given a question, ground-truth answer, and a candidate answer. Your task is to determine whether the candidate answer is semantically similar to the ground-truth.

Question: {question}
Ground-truth answer: {gt}
Candidate answer: {answer}

Please answer in one word: Yes or No.
"""

YES_KEYWORDS = ["yes", "Yes", "YES"]

def calculate_all_metrics(response_root, model_id):
    score = 0
    total = 0
    failed = 0

    for file in tqdm.tqdm(os.listdir(response_root)):
        with open(os.path.join(response_root, file), 'r') as f:
            saved_results = json.load(f)

        for _, value in saved_results.items():
            response = value.get('response', '')
            question = value['question']
            gt = value['answer']

            # 提取回答部分
            if 'The answer is:' in response:
                answer = response.split('The answer is:')[-1]
            elif 'The answer to the question is' in response:
                answer = response.split('The answer to the question is')[-1]
            else:
                answer = response

            try:
                prompt = _PROMPT.format(question=question.strip(), gt=gt.strip(), answer=answer.strip())

                result = openai.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                reply = result.choices[0].message.content.strip()

                if any(y in reply for y in YES_KEYWORDS):
                    score += 1
                total += 1
            except Exception as e:
                failed += 1
                total += 1

    print("----------- Evaluation Result -----------")
    print(f"Metric (Yes Rate): {score / total:.4f}")
    print(f"Failed Samples: {failed}")
    print(f"Total Samples: {total}")

image = re.compile(r'Image \b([0-9]|1[0-5])\b', flags=re.IGNORECASE)
image_2 = re.compile(r"'Image': \b([0-9]|1[0-5])\b", flags=re.IGNORECASE)

def acc_top_1(response_root):
    correct, total, failed = 0, 0, 0

    for file in os.listdir(response_root):
        with open(os.path.join(response_root, file), 'r') as f:
            saved_results = json.load(f)

        for _, value in saved_results.items():
            raw_resp = value['response'].split('The answer is:')[0]
            gt_indices = value['referred_figures_indices']

            try:
                img_pred = ast.literal_eval(raw_resp)['Image']
            except:
                try:
                    img_pred = int(image.findall(raw_resp)[0])
                except:
                    try:
                        img_pred = int(image_2.findall(raw_resp)[0])
                    except:
                        failed += 1
                        total += 1
                        continue

            if img_pred in gt_indices:
                correct += 1
            total += 1

    print("----------- Image Retrieval Accuracy -----------")
    print(f"Accuracy: {correct / total:.4f}")
    print(f"Failed Samples: {failed}")
    print(f"Total Samples: {total}")

# ---------------- Run ----------------
if __name__ == "__main__":
    calculate_all_metrics(args.response_root, args.model_id)
    acc_top_1(args.response_root)
