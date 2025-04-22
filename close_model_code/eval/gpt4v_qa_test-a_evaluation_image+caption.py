import time
import json
import random
import argparse
import base64
import os
import glob
import openai
import traceback

import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# ----------- Argument Parsing -----------
parser = argparse.ArgumentParser(description='Evaluate SPIQA with query-aware BGE retrieval.')
parser.add_argument('--response_root', type=str, required=True, help='Response Root path.')
parser.add_argument('--image_resolution', type=int, default=-1, help='Image Resolution.')
parser.add_argument('--model_id', type=str, required=True, help='gpt-4o / gpt-4o-mini')
parser.add_argument('--embedding_file', type=str, required=True, help='Path to JSONL BGE embedding file')
parser.add_argument('--top_k', type=int, default=8, help='Top-K retrieved chunks per query')
args = parser.parse_args()

# ----------- Paths -----------
if args.image_resolution == -1:
    _testA_IMAGE_ROOT = "/home/ec2-user/SPIQA/test-A/SPIQA_testA_Images"
else:
    raise NotImplementedError

with open('/home/ec2-user/SPIQA/test-A/SPIQA_testA.json', "r") as f:
    testA_data = json.load(f)

openai.api_key = ""
openai.base_url = "https://cmu.litellm.ai"

# ----------- Load BGE Segment Embeddings -----------
def load_embeddings_by_paper(jsonl_path):
    paper2segments = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            pid = obj["id"].split("_")[0]
            text = obj["text"]
            emb = np.array(obj["embedding"])
            paper2segments.setdefault(pid, []).append((text, emb))
    return paper2segments

paper2segments = load_embeddings_by_paper(args.embedding_file)
bge_model = BGEM3FlagModel("BAAI/bge-m3")

# ----------- Encode Image to Base64 -----------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ----------- Get Top-K Relevant Chunks -----------
def retrieve_top_k_chunks(paper_id, question, top_k):
    key = paper_id + ".txt"
    if key not in paper2segments:
        return ""
    segments = paper2segments[key]

    # Encode query vector
    result = bge_model.encode(question)
    if isinstance(result, dict):
        query_vec = next(iter(result.values()))
    else:
        query_vec = result

    # Compute cosine similarities
    scores = [
        float(np.dot(query_vec, seg_emb) / (np.linalg.norm(query_vec) * np.linalg.norm(seg_emb)))
        for _, seg_emb in segments
    ]
    topk_idx = np.argsort(scores)[-top_k:][::-1]
    topk_chunks = [segments[i][0] for i in topk_idx]
    return "\n".join(topk_chunks)

# ----------- Prompt Format -----------
def generate_prompt(question, retrieved_chunks):
    return (
        f"You are given a question, a few input images, and a caption corresponding to each input image.\n"
        f"Here is the paper text (split into short segments):\n"
        f"{retrieved_chunks}\n\n"
        f"Please answer the question based on the input images, captions, and the above text.\n"
        f"Question: {question}\n"
        f"Output in the following format: {{'Answer': 'Direct Answer to the Question'}}.\n"
    )

# ----------- Prepare Image Content -----------
def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures'].keys())
    referred_figures = [paper['qa'][question_idx]['reference']]
    answer = paper['qa'][question_idx]['answer']
    all_figures_captions = []

    if len(all_figures) > 8:
        others = list(set(all_figures) - set(referred_figures))
        random.shuffle(others)
        all_figures_modified = others[:8 - len(referred_figures)] + referred_figures
        random.shuffle(all_figures_modified)
    else:
        all_figures_modified = all_figures
        random.shuffle(all_figures_modified)

    referred_figures_indices = [all_figures_modified.index(ref) for ref in referred_figures]

    for figure in all_figures_modified:
        all_figures_captions.append(paper['all_figures'][figure]['caption'])

    all_figures_encoded = {}
    for idx, fig_name in enumerate(all_figures_modified):
        fig_path = os.path.join(_testA_IMAGE_ROOT, paper['paper_id'], fig_name)
        all_figures_encoded[f'figure_{idx}'] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(fig_path)}"}
        }

    return answer, all_figures_captions, all_figures_encoded, referred_figures_indices, all_figures_modified, referred_figures

# ----------- Main Inference -----------
def infer_gpt4v(testA_data, args):
    os.makedirs(args.response_root, exist_ok=True)

    for paper_id, paper in tqdm(testA_data.items(), desc="Processing papers"):
        output_file = os.path.join(args.response_root, f"{paper_id}_response.json")
        if os.path.exists(output_file):
            continue

        response_paper = {}

        try:
            for question_idx, qa in enumerate(paper['qa']):
                question = qa['question']
                answer, captions, images, ref_idx, figs, ref_figs = prepare_inputs(paper, question_idx)
                figure_type = paper['all_figures'][ref_figs[0]]['figure_type']
                content_type = paper['all_figures'][ref_figs[0]]['content_type']

                retrieved_chunks = retrieve_top_k_chunks(paper_id, question, top_k=args.top_k)
                prompt = generate_prompt(question, retrieved_chunks)
                print(prompt)
                input_prompt = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }

                for idx, key in enumerate(images.keys()):
                    input_prompt['messages'][0]['content'].append({"type": "text", "text": f"Image {idx}:"})
                    input_prompt['messages'][0]['content'].append(images[key])
                    input_prompt['messages'][0]['content'].append({"type": "text", "text": f"Caption {idx}: {captions[idx]} \n"})

                time.sleep(2)
                response = openai.chat.completions.create(
                    model=args.model_id,
                    messages=input_prompt['messages'],
                    max_tokens=256,
                )

                response_text = response.choices[0].message.content
                print(f"[{paper_id}] Q{question_idx}: {response_text.strip()}")

                response_paper[question_idx] = {
                    'question': question,
                    'referred_figures_indices': ref_idx,
                    'response': response_text,
                    'all_figures_names': figs,
                    'referred_figures_names': ref_figs,
                    'answer': answer,
                    'content_type': content_type,
                    'figure_type': figure_type
                }

        except Exception:
            traceback.print_exc()
            continue

        with open(output_file, 'w') as f:
            json.dump(response_paper, f)

# ----------- Entrypoint -----------
if __name__ == '__main__':
    infer_gpt4v(testA_data, args)
    print(f"Total responses saved: {len(glob.glob(args.response_root + '/*.json'))}")
