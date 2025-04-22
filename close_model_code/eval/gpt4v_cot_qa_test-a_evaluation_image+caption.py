import time
import json
import random
import argparse
import base64
import os
import glob
import traceback
import openai
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description='Evaluate with image+caption+retrieved text')
parser.add_argument('--response_root', type=str, required=True)
parser.add_argument('--image_resolution', type=int, default=-1)
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--embedding_file', type=str, required=True)
parser.add_argument('--top_k', type=int, default=8)
args = parser.parse_args()

# ---------------------- Paths ----------------------
if args.image_resolution == -1:
    _testA_IMAGE_ROOT = "/home/ec2-user/SPIQA/test-A/SPIQA_testA_Images"
else:
    raise NotImplementedError

with open('/home/ec2-user/SPIQA/test-A/SPIQA_testA.json', "r") as f:
    testA_data = json.load(f)

openai.api_key = ""
openai.base_url = "https://cmu.litellm.ai"

# ---------------------- Load Embeddings ----------------------
def extract_vector(raw):
    if isinstance(raw, dict):
        if "embedding" in raw: return np.array(raw["embedding"])
        if "dense_vecs" in raw: return np.array(raw["dense_vecs"])
        return np.array(next(iter(raw.values())))
    return np.array(raw)

def load_embeddings_by_paper(jsonl_path):
    paper2segments = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            pid = obj["id"].split("_")[0]
            vec = extract_vector(obj["embedding"])
            paper2segments.setdefault(pid, []).append((obj["text"], vec))
    return paper2segments

paper2segments = load_embeddings_by_paper(args.embedding_file)
bge_model = BGEM3FlagModel("BAAI/bge-m3")

# ---------------------- Utilities ----------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def retrieve_top_k_chunks(paper_id, question, top_k):
    key = paper_id + ".txt"
    if key not in paper2segments:
        return [], []
    segments = paper2segments[key]
    query_vec = extract_vector(bge_model.encode(question))  # extract from dict
    scores = [float(np.dot(query_vec, seg_emb) / (np.linalg.norm(query_vec) * np.linalg.norm(seg_emb))) for _, seg_emb in segments]
    topk_idx = np.argsort(scores)[-top_k:][::-1]
    topk_chunks = [segments[i][0] for i in topk_idx]
    return topk_chunks, topk_idx.tolist()

def build_prompt(question, text_chunks):
    joined_chunks = "\n".join([f"[{i}] {chunk}" for i, chunk in enumerate(text_chunks)])
    return (
        "You are given a question, a few input images, and a caption corresponding to each input image.\n"
        "You are also given some text chunks retrieved from the paper.\n"
        "Your task is to:\n"
        "1. Select the most helpful image+caption pair.\n"
        "2. Select the most relevant text chunk.\n"
        "3. Explain your rationale briefly.\n"
        "4. Answer the question directly.\n\n"
        f"Text Chunks:\n{joined_chunks}\n\n"
        "Format your output strictly as:\n"
        "{'Image': A, 'Text': B, 'Rationale': '...', 'Answer': '...'}\n\n"
        f"Question: {question}\n"
    )

def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures'].keys())
    referred = [paper['qa'][question_idx]['reference']]
    answer = paper['qa'][question_idx]['answer']
    all_captions, encoded_images = [], []

    if len(all_figures) > 8:
        others = list(set(all_figures) - set(referred))
        random.shuffle(others)
        all_figures = others[:8 - len(referred)] + referred
        random.shuffle(all_figures)

    ref_indices = [all_figures.index(r) for r in referred]
    for fig in all_figures:
        caption = paper['all_figures'][fig]['caption']
        all_captions.append(caption)
        img_path = os.path.join(_testA_IMAGE_ROOT, paper['paper_id'], fig)
        encoded = encode_image(img_path)
        encoded_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"}
        })

    return answer, all_captions, encoded_images, ref_indices, all_figures, referred

# ---------------------- Main Inference ----------------------
def infer(testA_data, args):
    os.makedirs(args.response_root, exist_ok=True)
    for paper_id, paper in tqdm(testA_data.items(), desc="Evaluating"):
        output_file = os.path.join(args.response_root, f"{paper_id}_response.json")
        if os.path.exists(output_file):
            continue

        response_paper = {}
        try:
            for q_idx, qa in enumerate(paper['qa']):
                question = qa['question']
                answer, captions, images, ref_idx, fig_names, ref_names = prepare_inputs(paper, q_idx)
                text_chunks, topk_ids = retrieve_top_k_chunks(paper_id, question, args.top_k)
                prompt = build_prompt(question, text_chunks)
                print(prompt)
                input_msg = [{"type": "text", "text": prompt}]
                for idx, img in enumerate(images):
                    input_msg.append({"type": "text", "text": f"Image {idx}:"})
                    input_msg.append(img)
                    input_msg.append({"type": "text", "text": f"Caption {idx}: {captions[idx]}\n"})

                response = openai.chat.completions.create(
                    model=args.model_id,
                    messages=[{"role": "user", "content": input_msg}],
                    max_tokens=512,
                )

                response_text = response.choices[0].message.content
                print(f"[{paper_id} Q{q_idx}] {response_text.strip()}")

                response_paper[q_idx] = {
                    'question': question,
                    'referred_figures_indices': ref_idx,
                    'response': response_text,
                    'all_figures_names': fig_names,
                    'referred_figures_names': ref_names,
                    'answer': answer,
                    'figure_type': paper['all_figures'][ref_names[0]]['figure_type'],
                    'content_type': paper['all_figures'][ref_names[0]]['content_type'],
                    'text_chunks': text_chunks,
                    'referred_text_chunks': [text_chunks[0]] if text_chunks else []
                }

        except Exception:
            traceback.print_exc()
            continue

        with open(output_file, "w") as f:
            json.dump(response_paper, f)

# ---------------------- Entrypoint ----------------------
if __name__ == '__main__':
    infer(testA_data, args)
    print(f"Saved {len(glob.glob(args.response_root + '/*.json'))} responses.")
