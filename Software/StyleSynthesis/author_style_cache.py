"""
author_style_cache.py

Answers: "can't we reuse features already computed by the trained CRNN,
and cache them per author, instead of recomputing everything every time?"

Short version: yes for the expensive part, not yet for the whole pipeline --
here's the honest split.

WHAT GETS CACHED, ONCE, PER AUTHOR (see build_and_cache_author_style below):
  1. The geometric style params from style_extraction.py (slant/x-height/
     spacing) -- cheap to compute, but there's no reason to redo the
     segmentation + measurement pass every time someone wants to generate
     text in this author's style, so it's cached anyway.
  2. A 128-d "style embedding" pooled from PaperCRNN's OWN convolutional
     backbone (train_paper_cnn_bilstm_ctc.py) -- the expensive part. Running
     every training-page line crop through 12 conv layers is exactly the
     kind of work that should happen once at enrollment time, not on every
     write request. This IS the technique from "Encoding CNN Activations for
     Writer Recognition" (arXiv:1712.07923): reuse activations from a CNN
     that was trained for a different task (here, content recognition, not
     writer ID) as a style/writer descriptor, because the conv layers still
     pick up local stroke-shape statistics as a side effect of learning to
     read handwriting at all. No changes needed to train_paper_cnn_bilstm_
     ctc.py -- PaperCRNN's stage1/pool1/stage2/pool2/stage3/height_pool
     submodules are called directly from here, bypassing the LSTM+CTC head,
     which is specific to reading left-to-right and not what we want.

WHAT THE CACHED EMBEDDING IS ACTUALLY USED FOR RIGHT NOW: nearest-neighbour
WRITER IDENTIFICATION (cosine similarity against every stored author's
embedding) -- this is buildable and useful today with the one author you
have data for, and it's literally your proposal's Requirement 2.

WHAT IT IS *NOT* USED FOR YET: driving glyph_styler.py's deformation
directly. Turning a 128-d embedding into "how much slant, how much
spacing, how much wobble" needs a learned mapping (embedding -> style
params), and that mapping has to be trained across MULTIPLE authors so it
generalizes instead of memorizing one person -- can't be done responsibly
on 1 author's data. The embedding is still cached and stored now so that
the day you have several authors' pages labelled, plugging in that learned
mapping is a one-function swap in glyph_styler.py, not a rewrite of this
caching layer.
"""

import json
import os

import numpy as np

from style_extraction import AuthorStyle, extract_author_style


def _cnn_backbone_forward(model, x):
    """Runs PaperCRNN's conv stack ONLY (no LSTM/CTC head) -- these are the
    same nn.Module attributes train_paper_cnn_bilstm_ctc.py's PaperCRNN.
    forward() calls, just stopped one step earlier. Returns (B, 128, 1, W)."""
    x = model.pool1(model.stage1(x))
    x = model.pool2(model.stage2(x))
    x = model.stage3(x)
    x = model.height_pool(x)
    return x


def extract_cnn_style_embedding(model, lineCropImages, device):
    """lineCropImages: list of PIL images (raw TEXT-line crops, as produced
    by NonDatasetSegmenterFP -- same crops classify_page.py already feeds
    the model for recognition). Returns a single 128-d numpy vector: mean-
    pooled over every timestep (width) of every line, i.e. one descriptor
    for the whole author, not per-character -- deliberately coarse, since
    at this stage we just want "does this reliably identify the author",
    not per-glyph detail."""
    import torch
    from train_paper_cnn_bilstm_ctc import resize_line_image_fixed, tensor_from_resized

    model.eval()
    vectors = []
    with torch.no_grad():
        for img in lineCropImages:
            resized = resize_line_image_fixed(img.convert("L"))
            tensor = tensor_from_resized(resized).unsqueeze(0).to(device)
            feats = _cnn_backbone_forward(model, tensor)   # (1, 128, 1, W)
            pooled = feats.mean(dim=(0, 2, 3))              # (128,) -- average over width too
            vectors.append(pooled.cpu().numpy())

    if not vectors:
        raise ValueError("No line crops given -- nothing to embed.")
    return np.mean(np.stack(vectors, axis=0), axis=0)


def build_and_cache_author_style(authorId, imgPath, weightsPath, cacheDir, computeCnnEmbedding=True):
    """One-time (well -- one-time-per-new-training-page) expensive step.
    Segments the page, measures geometric style, optionally runs the CNN
    embedding, and writes everything to cacheDir/<authorId>.npz +
    cacheDir/<authorId>.json (geometric params as human-readable JSON,
    embedding as compact binary npz -- so you can eyeball the geometric
    numbers without needing numpy)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN"))
    import NonDatasetSegmenterFP as Segmenter

    results, preview, meta = Segmenter.ProcessPage(imgPath)
    style = extract_author_style(results, meta)

    embedding = None
    if computeCnnEmbedding:
        import torch
        from PIL import Image
        from train_paper_cnn_bilstm_ctc import CHARSET, PaperCRNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
        state_dict = torch.load(weightsPath, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        lineCrops = [Image.fromarray(r["raw_crop"]) for r in results if r["tag"] == "TEXT"]
        embedding = extract_cnn_style_embedding(model, lineCrops, device)

    os.makedirs(cacheDir, exist_ok=True)
    jsonPath = os.path.join(cacheDir, f"{authorId}.json")
    with open(jsonPath, "w") as f:
        json.dump({
            "authorId": authorId,
            "slantDeg": style.slantDeg,
            "xHeightPx": style.xHeightPx,
            "spacingRatio": style.spacingRatio,
            "nLinesUsed": style.nLinesUsed,
            "hasEmbedding": embedding is not None,
        }, f, indent=2)

    if embedding is not None:
        npzPath = os.path.join(cacheDir, f"{authorId}.npz")
        np.savez(npzPath, embedding=embedding)

    print(f"Cached style for '{authorId}' -> {jsonPath}" + (f" + {npzPath}" if embedding is not None else ""))
    return style, embedding


def load_author_style(authorId, cacheDir):
    """The cheap path -- no segmentation, no CNN forward pass, just a file
    read. This is what glyph_styler.py should call at write-time."""
    jsonPath = os.path.join(cacheDir, f"{authorId}.json")
    with open(jsonPath) as f:
        data = json.load(f)
    style = AuthorStyle(data["slantDeg"], data["xHeightPx"], data["spacingRatio"], data["nLinesUsed"])

    embedding = None
    npzPath = os.path.join(cacheDir, f"{authorId}.npz")
    if os.path.exists(npzPath):
        embedding = np.load(npzPath)["embedding"]

    return style, embedding


def identify_author(queryEmbedding, cacheDir):
    """Nearest-neighbour writer ID (Requirement 2 of your proposal) via
    cosine similarity against every cached author embedding -- the concrete,
    usable-today thing the CNN embedding buys you, independent of whether
    it ever ends up driving glyph deformation."""
    candidates = []
    for fname in os.listdir(cacheDir):
        if not fname.endswith(".npz"):
            continue
        authorId = fname[:-4]
        embedding = np.load(os.path.join(cacheDir, fname))["embedding"]
        cosine = float(np.dot(queryEmbedding, embedding) /
                        (np.linalg.norm(queryEmbedding) * np.linalg.norm(embedding) + 1e-8))
        candidates.append((authorId, cosine))

    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates  # [(authorId, similarity), ...] best match first


if __name__ == "__main__":
    defaultImg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                               "NOGIT", "NonDatasetImages", "baseline_model.png")
    defaultWeights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                                   "NOGIT", "weights", "paper_cnn_bilstm_ctc_best.pt")
    cacheDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN", "NOGIT", "AuthorStyles")

    authorId = input("Author id to cache this page under (blank = 'dylan'): ").strip() or "dylan"
    imgPath = input(f"Training page (blank = {os.path.basename(defaultImg)}): ").strip() or defaultImg

    doCnn = input("Also compute the CNN embedding? Needs torch + trained weights [y/N]: ").strip().lower() == "y"
    weightsPath = defaultWeights
    if doCnn:
        weightsPath = input(f"Weights path (blank = {os.path.basename(defaultWeights)}): ").strip() or defaultWeights

    style, embedding = build_and_cache_author_style(authorId, imgPath, weightsPath, cacheDir, computeCnnEmbedding=doCnn)
    print(f"\nGeometric style: {style}")
    if embedding is not None:
        print(f"CNN embedding: shape {embedding.shape}, first 5 values {embedding[:5]}")

    print(f"\nNext time you want this author's style, just call load_author_style('{authorId}', cacheDir) --"
          f" no re-segmentation, no re-running the CNN.")
