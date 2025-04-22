# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0
# …

import os
import json
import random
import math
import argparse
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ── Paths ───────────────────────────────────────────────────────────────────────
TESTA_JSON       = '../../../datasets/test-A/SPIQA_testA.json'
TESTA_IMAGE_ROOT = '../../../datasets/test-A/SPIQA_testA_Images_224px'

with open(TESTA_JSON, 'r') as f:
    TESTA_DATA = json.load(f)

# ── Prompt templates ─────────────────────────────────────────────────────────────
PROMPT_SELECT = (
    "Caption: {caption}\n"
    "Question: {question}\n"
    "Is this figure helpful to answer the question? Answer Yes or No."
)

PROMPT_ANSWER = (
    "Below are the captions of the figures selected as helpful:\n\n"
    "{selected_captions}\n\n"
    "Question: {question}\n"
    "Please provide a concise answer."
)

# ── Utility to tile a handful of images into one single PIL montage ───────────────
def make_montage(images, thumb_size=(224,224), max_images=4):
    imgs = images[:max_images]
    n    = len(imgs)
    cols = min(n, 2)
    rows = math.ceil(n/cols)
    W, H = cols*thumb_size[0], rows*thumb_size[1]
    canvas = Image.new('RGB', (W, H), 'white')
    for i,img in enumerate(imgs):
        r,c = divmod(i, cols)
        canvas.paste(img.resize(thumb_size), (c*thumb_size[0], r*thumb_size[1]))
    return canvas

def infer_testA(data, args):
    proc  = InstructBlipProcessor.from_pretrained(args.model_id, use_fast=False)
    model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_id,
                load_in_4bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    model.eval()

    os.makedirs(args.response_root, exist_ok=True)

    for paper_id, paper in sorted(data.items(), key=lambda x: random.random()):
        out_path = os.path.join(args.response_root, f"{paper_id}_response.json")
        if os.path.exists(out_path):
            continue

        all_figs = list(paper['all_figures'].keys())
        caps     = {f: paper['all_figures'][f]['caption'] for f in all_figs}
        result   = {}

        for qidx, qa in enumerate(paper['qa']):
            question    = qa['question']
            ground      = qa['answer']
            fallback    = qa.get('reference', all_figs[:1])[0]

            # ── Stage 1: filter all figures ─────────────────────────
            selected = []
            for fig in all_figs:
                prompt1 = PROMPT_SELECT.format(
                    caption=caps[fig],
                    question=question
                )
                img = Image.open(os.path.join(TESTA_IMAGE_ROOT, paper_id, fig)).convert('RGB')
                img = img.resize((args.image_resolution, args.image_resolution))

                inputs = proc(images=img, text=prompt1, return_tensors="pt")
                inputs = {k:v.to(model.device) for k,v in inputs.items()}
                outs   = model.generate(
                            **inputs,
                            num_beams=2,
                            max_new_tokens=16,
                            temperature=0.0,
                            do_sample=False
                        )
                reply = proc.batch_decode(outs, skip_special_tokens=True)[0].lower()
                if reply.startswith("yes"):
                    selected.append(fig)

            if not selected:
                selected = [fallback]

            # ── Stage 2: one‐shot answer ────────────────────────────
            sel_caps     = "\n".join(f"{f}: {caps[f]}" for f in selected)
            answer_prom  = PROMPT_ANSWER.format(
                                selected_captions=sel_caps,
                                question=question
                            )

            # build a single montage of the chosen figures
            imgs = []
            for f in selected:
                img = Image.open(os.path.join(TESTA_IMAGE_ROOT, paper_id, f)).convert('RGB')
                imgs.append(img)
            montage = make_montage(imgs)

            inputs = proc(
                images=montage,
                text=answer_prom,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            inputs = {k:v.to(model.device) for k,v in inputs.items()}
            outs   = model.generate(
                        **inputs,
                        num_beams=5,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
            answer = proc.batch_decode(outs, skip_special_tokens=True)[0].strip()

            result[qidx] = {
                "question":         question,
                "ground_truth":     ground,
                "selected_figures": selected,
                "answer":           answer
            }

        with open(out_path, 'w') as wf:
            json.dump(result, wf, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test‑A improved inference")
    parser.add_argument('--model_id',        type=str,
                        default="Salesforce/instructblip-vicuna-7b")
    parser.add_argument('--response_root',   type=str, required=True,
                        help="Where to write per-paper JSON")
    parser.add_argument('--image_resolution',type=int, required=True,
                        help="Must be 224")
    args = parser.parse_args()

    infer_testA(TESTA_DATA, args)