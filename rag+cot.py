# instructblip_rag_cot.py
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 …

import os
import json
import math
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from FlagEmbedding import BGEM3FlagModel

# ── Paths ───────────────────────────────────────────────────────────────────────
SPIQA_JSON = '/home/ubuntu/spiqa/datasets/test-A/SPIQA_testA.json'
IMG_ROOT   = '/home/ubuntu/spiqa/datasets/test-A/SPIQA_testA_Images_224px'

# ── Device ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Retrieval index loader ─────────────────────────────────────────────────────
def extract_vector(raw):
    if isinstance(raw, dict):
        if "embedding" in raw:    return np.array(raw["embedding"])
        if "dense_vecs" in raw:   return np.array(raw["dense_vecs"])
        return np.array(next(iter(raw.values())))
    return np.array(raw)

def load_embeddings_by_paper(jsonl_path):
    paper2segments = {}
    with open(jsonl_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSONL line {lineno}: {e}")
                continue
            pid = obj.get("id", "").split("_")[0]
            vec = extract_vector(obj.get("embedding", {}))
            text = obj.get("text", "")
            paper2segments.setdefault(pid, []).append((text, vec))
    return paper2segments

# ── Retrieval ───────────────────────────────────────────────────────────────────
def retrieve_top_k_chunks(paper2segments, bge_model, paper_id, question, top_k):
    key = paper_id + ".txt"
    if key not in paper2segments:
        return []
    segments = paper2segments[key]
    # encode with BGE and extract raw vector
    raw_q = bge_model.encode(question)
    qvec = torch.tensor(extract_vector(raw_q), dtype=torch.float32, device=device)
    qnorm = qvec.norm()
    sims = []
    for _, seg_emb in segments:
        t = torch.tensor(seg_emb, dtype=torch.float32, device=device)
        sims.append((qvec @ t) / (qnorm * t.norm()))
    sims = torch.stack(sims).cpu().numpy()
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [segments[i][0] for i in topk_idx]

# ── Prompt templates ─────────────────────────────────────────────────────────────
COT_SELECT = """
Relevant text snippets from the paper:
{chunks}

Below are all figures from the paper (tiled together) with captions:
{all_list}

Question: {question}

Step 1: Which figure filename(s) will help answer this?
List filenames separated by semicolons, e.g.:
fig1.png; fig3.png
"""

COT_RATIONALE = """
You selected: {sel_figs}

Question: {question}

Step 2: In one or two sentences, explain how these figure(s) and text support the answer.
"""

COT_ANSWER = """
Based on the above reasoning, provide a one-sentence final answer.
Question: {question}
"""

# ── Montage utility ──────────────────────────────────────────────────────────────
def make_montage(images, thumb_size=(128,128), max_images=16):
    imgs = images[:max_images]
    n    = max(1, len(imgs))
    cols = min(4, n)
    rows = math.ceil(n/cols)
    W, H = cols*thumb_size[0], rows*thumb_size[1]
    canvas = Image.new('RGB', (W, H), 'white')
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(img.resize(thumb_size), (c*thumb_size[0], r*thumb_size[1]))
    return canvas

# ── Main inference ──────────────────────────────────────────────────────────────
def infer_with_rag_cot(testA_data, embeddings, args):
    assert args.image_resolution == 224, "Only 224px supported"
    os.makedirs(args.response_root, exist_ok=True)

    # load retriever (stays on CPU if no .to())
    bge_model = BGEM3FlagModel("BAAI/bge-m3")

    # load InstructBlip on GPU
    proc  = InstructBlipProcessor.from_pretrained(args.model_id, use_fast=False)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    for paper_id, paper in tqdm(testA_data.items(), desc="Papers"):
        out_path = os.path.join(args.response_root, f"{paper_id}_response.json")
        if os.path.exists(out_path):
            continue

        all_figs = list(paper['all_figures'].keys())
        all_caps = [paper['all_figures'][f]['caption'] for f in all_figs]
        cap_block = "\n".join(f"{f}: {c}" for f,c in zip(all_figs, all_caps))
        if len(cap_block)>2000:
            cap_block = cap_block[:2000] + "…"

        imgs = []
        for f in all_figs:
            pth = os.path.join(IMG_ROOT, paper_id, f)
            try:
                im = Image.open(pth).convert('RGB')
            except:
                im = Image.new('RGB', (224,224), 'white')
            imgs.append(im)
        montage_all = make_montage(imgs)

        resp = {}
        for qidx, qa in enumerate(paper['qa']):
            question, gt = qa['question'], qa['answer']

            # Retrieval
            chunks      = retrieve_top_k_chunks(embeddings, bge_model, paper_id, question, args.top_k)
            chunk_block = "\n".join(f"[{i}] {c}" for i,c in enumerate(chunks))

            # Step 1: select figures
            sel_prompt = COT_SELECT.format(chunks=chunk_block, all_list=cap_block, question=question)
            inp = proc(images=montage_all, text=sel_prompt,
                       return_tensors="pt", padding="max_length",
                       truncation=True, max_length=512)
            inp = {k:v.to(model.device) for k,v in inp.items()}
            out = model.generate(**inp,
                                 num_beams=3, max_new_tokens=64,
                                 temperature=0.0, do_sample=False)
            sel_txt = proc.batch_decode(out, skip_special_tokens=True)[0]
            sel_figs = [p.strip()
                        for ln in sel_txt.splitlines()
                        for p in ln.replace("\n",";").split(";")
                        if p.strip() in all_figs]
            if not sel_figs:
                sel_figs = [all_figs[0]]

            # montage of selected
            sel_imgs = []
            for f in sel_figs:
                pth = os.path.join(IMG_ROOT, paper_id, f)
                try:  im = Image.open(pth).convert('RGB')
                except: im = Image.new('RGB', (224,224), 'white')
                sel_imgs.append(im)
            montage_sel = make_montage(sel_imgs)

            # Step 2: rationale
            rat_prompt = COT_RATIONALE.format(sel_figs="; ".join(sel_figs), question=question)
            inp = proc(images=montage_sel, text=rat_prompt,
                       return_tensors="pt", padding="max_length",
                       truncation=True, max_length=512)
            inp = {k:v.to(model.device) for k,v in inp.items()}
            out = model.generate(**inp,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 min_new_tokens=10, max_new_tokens=64,
                                 repetition_penalty=1.2)
            rationale = proc.batch_decode(out, skip_special_tokens=True)[0].strip()

            # Step 3: answer
            ans_prompt = COT_ANSWER.format(question=question)
            inp = proc(images=montage_sel, text=ans_prompt,
                       return_tensors="pt", padding="max_length",
                       truncation=True, max_length=512)
            inp = {k:v.to(model.device) for k,v in inp.items()}
            out = model.generate(**inp,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 min_new_tokens=5, max_new_tokens=32,
                                 repetition_penalty=1.2)
            final_ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip()

            resp[qidx] = {
                "question":         question,
                "ground_truth":     gt,
                "selected_figures": sel_figs,
                "retrieved_texts":  chunks,
                "rationale":        rationale,
                "answer":           final_ans
            }

        with open(out_path, 'w') as wf:
            json.dump(resp, wf, indent=2)


if __name__=="__main__":
    p = argparse.ArgumentParser(description="RAG + 3‑step CoT w/ InstructBLIP")
    p.add_argument('--model_id',        type=str,   required=True)
    p.add_argument('--response_root',   type=str,   required=True)
    p.add_argument('--image_resolution',type=int,   required=True)
    p.add_argument('--embedding_file',  type=str,   required=True)
    p.add_argument('--top_k',           type=int,   default=8)
    args = p.parse_args()

    with open(SPIQA_JSON, 'r') as rf:
        TESTA = json.load(rf)
    embeddings = load_embeddings_by_paper(args.embedding_file)

    infer_with_rag_cot(TESTA, embeddings, args)