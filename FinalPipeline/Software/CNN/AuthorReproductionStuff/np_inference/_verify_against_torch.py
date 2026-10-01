"""Dev-only comparison tool: runs the numpy inference path and the existing
torch-based path on the same real sample images and checks they agree.
Imports torch on purpose -- this file is NOT part of the inference path,
just a correctness check, and is never imported from text_model.py /
author_model.py / weight_io.py / layers.py."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # AuthorReproductionStuff
CNN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CNN_DIR))

from TrainText import IAMLineDatasetRaw, _decode_png, PaperCRNN, CHARSET, decode_ctc as torch_decode_ctc, resize_line_image_fixed as torch_resize, tensor_from_resized as torch_tensor_from
from TrainAuthor import AuthorClassifierCNN
from TrainAuthorShape import StrokeNormalize as torch_StrokeNormalize
import VerifyRewrite as VR
import EvaluateStyle as ES

sys.path.insert(0, str(SCRIPT_DIR / "np_inference"))
import np_inference.text_model as ntm
import np_inference.author_model as nam


def main():
    device = torch.device("cpu")

    text_model_t = VR.LoadTextModel(device)
    text_model_np = ntm.PaperCRNNNumpy(checkpoint_path=VR.TEXT_WEIGHTS)
    assert str(VR.TEXT_WEIGHTS) == str(ntm.DEFAULT_TEXT_WEIGHTS), (VR.TEXT_WEIGHTS, ntm.DEFAULT_TEXT_WEIGHTS)

    author_model_t, mapping_t, idx_to_author = ES.LoadAuthorModel(device)
    author_model_np = nam.AuthorClassifierCNNNumpy(checkpoint_path=ES.AUTHOR_WEIGHTS)
    assert str(ES.AUTHOR_WEIGHTS) == str(nam.DEFAULT_AUTHOR_WEIGHTS)
    assert author_model_np.author_mapping == mapping_t

    data_dir = CNN_DIR.parent.parent / "Data" / "Datasets" / "IAMpages10"
    cache_dir = CNN_DIR / "NOGIT" / "line_cache_authors10"
    base = IAMLineDatasetRaw(root_dir=str(data_dir), cache_dir=str(cache_dir))
    print(f"[data] {len(base.samples)} total line samples")

    # sample a handful of lines spread across authors
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(base.samples), size=min(20, len(base.samples)), replace=False)

    text_matches, text_total = 0, 0
    author_matches, author_total = 0, 0
    max_logit_diff = 0.0
    max_text_logprob_diff = 0.0

    for i in idxs:
        s = base.samples[i]
        pil_img = _decode_png(s["image_png"])  # raw crop, exactly what ReadText/ClassifyImage take
        true_text = s["text"]
        true_author = s["page_key"].split("/")[0]

        # --- text: torch vs numpy ---
        pred_t = VR.ReadText(text_model_t, pil_img, device)
        pred_np = ntm.ReadText(pil_img, text_model_np)

        # also compare raw log-probs at a shared preprocessing point
        t_input = torch_tensor_from(torch_resize(pil_img)).unsqueeze(0)
        with torch.no_grad():
            logp_t = text_model_t(t_input).numpy()
        np_input = ntm.tensor_from_resized(ntm.resize_line_image_fixed(pil_img))
        logp_np = text_model_np.forward(np_input)
        diff = np.abs(logp_t - logp_np).max()
        max_text_logprob_diff = max(max_text_logprob_diff, float(diff))

        text_total += 1
        if pred_t == pred_np:
            text_matches += 1
        else:
            print(f"[TEXT MISMATCH] true={true_text!r} torch={pred_t!r} numpy={pred_np!r} logp_diff={diff:.2e}")

        # --- author: torch vs numpy ---
        norm_t = torch_StrokeNormalize(np.array(pil_img.convert("L")))
        img_for_author = norm_t if norm_t is not None else pil_img
        t_input_a = torch_tensor_from(torch_resize(img_for_author)).unsqueeze(0)
        with torch.no_grad():
            logits_t = author_model_t(t_input_a).numpy()
        pred_idx_t, probs_t = ES.ClassifyImage(author_model_t, pil_img, device)

        pred_idx_np, probs_np = nam.ClassifyImage(pil_img, author_model_np)
        norm_np_img = nam.StrokeNormalize(np.array(pil_img.convert("L")))
        img_for_author_np = norm_np_img if norm_np_img is not None else pil_img
        np_input_a = nam.tensor_from_resized(nam.resize_line_image_fixed(img_for_author_np))
        logits_np = author_model_np.forward(np_input_a)
        ldiff = np.abs(logits_t - logits_np).max()
        max_logit_diff = max(max_logit_diff, float(ldiff))

        author_total += 1
        pred_author_t = idx_to_author[pred_idx_t]
        pred_author_np = author_model_np.author_mapping and {v: k for k, v in author_model_np.author_mapping.items()}[pred_idx_np]
        if pred_idx_t == pred_idx_np:
            author_matches += 1
        else:
            print(f"[AUTHOR MISMATCH] true={true_author} torch={pred_author_t} numpy={pred_author_np} logit_diff={ldiff:.2e}")

    print()
    print(f"TEXT: {text_matches}/{text_total} exact decoded-string matches. max log-prob diff = {max_text_logprob_diff:.2e}")
    print(f"AUTHOR: {author_matches}/{author_total} exact predicted-class matches. max logit diff = {max_logit_diff:.2e}")


if __name__ == "__main__":
    main()
