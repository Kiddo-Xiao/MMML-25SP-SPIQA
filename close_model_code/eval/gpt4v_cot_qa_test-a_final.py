#!/usr/bin/env python
# -*- coding: utf‑8 -*-
# ------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------- #
import os, json, random, argparse, base64, glob, traceback
import numpy as np
from tqdm import tqdm
import openai
from FlagEmbedding import BGEM3FlagModel

# ------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--response_root", type=str, required=True)
parser.add_argument("--model_id",      type=str, required=True)
parser.add_argument("--embedding_file",type=str, required=True)
parser.add_argument("--top_k",         type=int, default=8)
parser.add_argument("--max_images",    type=int, default=4)
args = parser.parse_args()

# ------------------------------------------------------------------- #
# Paths / API keys
# ------------------------------------------------------------------- #
TESTA_JSON       = "/home/ec2-user/SPIQA/test-A/SPIQA_testA.json"
TESTA_IMAGE_ROOT = "/home/ec2-user/SPIQA/test-A/SPIQA_testA_Images"

openai.api_key  = os.getenv("OPENAI_API_KEY", "")
openai.base_url = "https://cmu.litellm.ai"

# ------------------------------------------------------------------- #
# Embedding helpers
# ------------------------------------------------------------------- #
def _vec(raw):
    """Robustly convert raw embedding item to a 1‑D float32 numpy array."""
    if isinstance(raw, dict):
        if "embedding"  in raw: return np.asarray(raw["embedding"],  dtype=np.float32)
        if "dense_vecs" in raw: return np.asarray(raw["dense_vecs"], dtype=np.float32)
        return np.asarray(next(iter(raw.values())), dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)

def load_embeds(jsonl):
    d = {}
    with open(jsonl) as f:
        for ln in f:
            obj = json.loads(ln)
            pid = obj["id"].split("_")[0]
            d.setdefault(pid, []).append((obj["text"], _vec(obj["embedding"])))
    return d

paper2segments = load_embeds(args.embedding_file)
bge_model      = BGEM3FlagModel("BAAI/bge-m3")

# ------------------------------------------------------------------- #
# Utilities
# ------------------------------------------------------------------- #
def encode_img(p):  return base64.b64encode(open(p,"rb").read()).decode()
def cosine(a,b):    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def retrieve(pid,q,k):
    key = pid + ".txt"
    if key not in paper2segments: return [], []
    segs, qv = paper2segments[key], _vec(bge_model.encode(q))
    idx = np.argsort([cosine(qv,v) for _,v in segs])[-k:][::-1]
    return [segs[i][0] for i in idx], idx.tolist()

def chat(msgs, model, mx=256, fmt=None):
    return openai.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":msgs}],
        response_format=fmt or {"type":"text"},
        temperature=0.0,
        max_tokens=mx
    ).choices[0].message.content.strip()

# ------------------------------------------------------------------- #
# Stage 1  caption‑only filter
# ------------------------------------------------------------------- #
SEL_TMPL = ("Below are figure captions.\n"
            "Select indices helpful for the QUESTION.\n"
            "Return only a JSON list e.g., [0,2] or [].\n\n{caps}\n\nQUESTION: {q}")

def choose_figs(q,caps):
    prompt = SEL_TMPL.format(caps="\n".join(f"[{i}] {c}" for i,c in enumerate(caps)),q=q)
    try:
        idx = json.loads(chat(prompt,args.model_id))
        if isinstance(idx,list) and all(isinstance(i,int) for i in idx):
            return [i for i in idx if 0<=i<len(caps)]
    except Exception: pass
    return list(range(len(caps)))

# ------------------------------------------------------------------- #
# Prepare inputs
# ------------------------------------------------------------------- #
def prep(paper,qi):
    figs    = list(paper["all_figures"].keys())
    ref_val = paper["qa"][qi].get("reference")
    fallback= ref_val[0] if isinstance(ref_val,list) else ref_val or figs[0]

    if len(figs)>8:
        others=list(set(figs)-{fallback}); random.shuffle(others)
        figs=others[:7]+[fallback]; random.shuffle(figs)

    caps,paths=[],[]
    for f in figs:
        caps.append(paper["all_figures"][f]["caption"])
        paths.append(os.path.join(TESTA_IMAGE_ROOT,paper["paper_id"],f))
    return paper["qa"][qi]["answer"],caps,paths,fallback,figs

# ------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------- #
def run(data):
    os.makedirs(args.response_root,exist_ok=True)
    for pid,paper in tqdm(data.items(),desc="Papers"):
        out=os.path.join(args.response_root,f"{pid}_response.json")
        if os.path.exists(out): continue
        res={}
        try:
            for qi,qa in enumerate(paper["qa"]):
                q=qa["question"]
                gt,caps,paths,fallback,figs=prep(paper,qi)
                sel=choose_figs(q,caps) or [figs.index(fallback)]
                sel=sel[:args.max_images]

                sel_caps=[caps[i] for i in sel]
                sel_paths=[paths[i] for i in sel]
                sel_names=[figs[i] for i in sel]
                chunks,_=retrieve(pid,q,args.top_k)
                chunk_blk="\n".join(f"[{i}] {c}" for i,c in enumerate(chunks))

                header=("You are given a question, several images (each with a caption), "
                        "and some retrieved text chunks.\n"
                        "Pick the most helpful image index and text chunk index, explain briefly, "
                        "then answer.\n"
                        "Return only a JSON object with keys "
                        "'Image','Text','Rationale','Answer'.\n\n"
                        f"Text Chunks:\n{chunk_blk}\n")
                mm=[{"type":"text","text":header}]
                for j,(cap,pth) in enumerate(zip(sel_caps,sel_paths)):
                    mm+=[{"type":"text","text":f"Image {j}:"},
                         {"type":"image_url",
                          "image_url":{"url":f"data:image/png;base64,{encode_img(pth)}"}},
                         {"type":"text","text":f"Caption {j}: {cap}\n"}]
                mm.append({"type":"text","text":f"Question: {q}"})

                raw=chat(mm,args.model_id,mx=512,fmt={"type":"json_object"})
                ans=json.loads(raw)

                res[qi]={ "question":q,"ground_truth":gt,
                          "selected_figures_names":sel_names,
                          "all_figures_names":figs,
                          "chosen_image_idx":ans.get("Image"),
                          "chosen_text_idx": ans.get("Text"),
                          "rationale":       ans.get("Rationale"),
                          "answer":          ans.get("Answer"),
                          "text_chunks":     chunks,
                          "raw_gpt":         raw }
        except Exception: traceback.print_exc(); continue
        with open(out,"w") as f: json.dump(res,f,indent=2)

# ------------------------------------------------------------------- #
if __name__=="__main__":
    with open(TESTA_JSON) as f: data=json.load(f)
    run(data)
    print(f"✔  Saved {len(glob.glob(os.path.join(args.response_root,'*.json')))} files.")
