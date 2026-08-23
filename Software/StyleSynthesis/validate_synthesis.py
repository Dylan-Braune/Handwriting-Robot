"""
validate_synthesis.py

The "confirm these mappings work incredibly accurately, before we even think
about motor movements" check, run entirely in software.

Two independent checks, because they answer two different questions and
your proposal actually specifies both separately (Requirement 1: 85%
character accuracy; Requirement 1: 85% stylistic correlation to the chosen
author):

1. CONTENT check -- does the synthesized word still say what it's supposed
   to say? Rasterizes the styled strokes, feeds the image through your
   ALREADY-TRAINED PaperCRNN (the exact same model/preprocessing
   classify_page.py uses), decodes it, and measures character accuracy
   against the text you asked it to write. This needs torch, so it only
   runs on a machine that has it installed (this dev sandbox doesn't).

2. STYLE self-consistency check -- runs today, no torch needed. Since you
   currently have full segmentation+labels for what looks like ONE author
   (see style_extraction.py's docstring), a true cross-author "does this
   look like author A and not author B" classifier check isn't buildable
   yet -- that needs >=2 authors' labelled data at minimum, and your
   proposal's own Requirement 6 wants 10. What IS checkable today: split
   that author's lines into two halves, extract style from each half
   independently, and confirm the two measurements agree with each other.
   If they don't, the style extraction itself is unstable and no amount of
   downstream deformation will look convincingly like "this author" --
   worth knowing before scaling up to more authors.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN"))

import NonDatasetSegmenterFP as Segmenter
from style_extraction import extract_author_style
from glyph_styler import style_text_to_strokes
from rasterize import rasterize_strokes


def check_style_self_consistency(imgPath):
    """Splits the page's TEXT lines into two halves and compares the style
    extracted from each -- a same-author-should-agree-with-itself check,
    achievable with just the one author's data you have right now."""
    results, preview, meta = Segmenter.ProcessPage(imgPath)
    textResults = [r for r in results if r["tag"] == "TEXT"]
    if len(textResults) < 4:
        print("Not enough TEXT lines on this page to split in half meaningfully (need >=4).")
        return None

    half = len(textResults) // 2
    firstHalf = [r for r in results if r["tag"] != "TEXT" or r in textResults[:half]]
    secondHalf = [r for r in results if r["tag"] != "TEXT" or r in textResults[half:]]

    styleA = extract_author_style(firstHalf, meta)
    styleB = extract_author_style(secondHalf, meta)

    print(f"Style from first half  ({styleA.nLinesUsed} lines): {styleA}")
    print(f"Style from second half ({styleB.nLinesUsed} lines): {styleB}")

    slantDiff = abs(styleA.slantDeg - styleB.slantDeg)
    xHeightDiff = abs(styleA.xHeightPx - styleB.xHeightPx) / max(styleA.xHeightPx, styleB.xHeightPx)
    spacingDiff = abs(styleA.spacingRatio - styleB.spacingRatio)

    print(f"\nDisagreement between the two halves:")
    print(f"  slant     : {slantDiff:.1f} deg   (want < ~3 deg for a stable estimate)")
    print(f"  x-height  : {xHeightDiff*100:.1f}% relative   (want < ~10%)")
    print(f"  spacing   : {spacingDiff:.3f} ratio units   (want < ~0.03)")

    stable = slantDiff < 3.0 and xHeightDiff < 0.10 and spacingDiff < 0.03
    print(f"\n{'STABLE' if stable else 'UNSTABLE'} -- "
          f"{'this author style measurement should generalize reasonably to new pages of theirs.' if stable else 'style measurement is noisy; consider more training pages before trusting the deformation.'}")
    return stable


def check_content_accuracy(imgPath, targetText, weightsPath):
    """Requires torch -- run this on your training/dev machine, not
    necessarily in a lightweight CI sandbox. Mirrors classify_page.py's
    model-loading and preprocessing exactly so this is a fair test of what
    the real deployed classifier would actually decode."""
    import torch
    from PIL import Image
    from train_paper_cnn_bilstm_ctc import (
        CHARSET, PaperCRNN, decode_ctc, levenshtein,
        resize_line_image_fixed, tensor_from_resized,
    )

    results, preview, meta = Segmenter.ProcessPage(imgPath)
    style = extract_author_style(results, meta)
    print(f"Style used for synthesis: {style}")

    strokes = style_text_to_strokes(targetText, style)
    img = rasterize_strokes(strokes, lineWidthPx=max(1, int(style.xHeightPx * 0.06)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    state_dict = torch.load(weightsPath, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    resized = resize_line_image_fixed(img.convert("L"))
    tensor = tensor_from_resized(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(tensor)
    predicted = decode_ctc(log_probs)[0]

    err = levenshtein(predicted, targetText)
    charAcc = 1 - err / max(1, len(targetText))

    print(f"\nTarget text : {targetText!r}")
    print(f"Predicted   : {predicted!r}")
    print(f"Char accuracy: {charAcc*100:.1f}%  ({err} edit(s) / {len(targetText)} chars)  "
          f"(spec: 85% per Requirement 1)")
    return charAcc


if __name__ == "__main__":
    defaultImg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                               "NOGIT", "NonDatasetImages", "baseline_model.png")
    imgPath = input(f"Training page (blank = {os.path.basename(defaultImg)}): ").strip() or defaultImg

    print("\n=== Style self-consistency check (no torch needed) ===")
    check_style_self_consistency(imgPath)

    print("\n=== Content accuracy check (requires torch + a trained model) ===")
    runIt = input("Run it? [y/N]: ").strip().lower()
    if runIt == "y":
        defaultWeights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                                       "NOGIT", "weights", "paper_cnn_bilstm_ctc_best.pt")
        weightsPath = input(f"Weights path (blank = {os.path.basename(defaultWeights)}): ").strip() or defaultWeights
        targetText = input("Text to synthesize + check (blank = 'this word was never written before'): ").strip() \
            or "this word was never written before"
        check_content_accuracy(imgPath, targetText, weightsPath)
    else:
        print("Skipped -- run this on a machine with torch + your trained weights when ready.")
