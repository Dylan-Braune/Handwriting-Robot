"""
glyph_styler.py

Applies an AuthorStyle (style_extraction.py) to base_alphabet.py's neutral
Hershey glyph skeletons, producing a styled stroke sequence for arbitrary
text -- including words/characters this author never wrote in training,
which is the entire point of factoring content (alphabet) apart from style
(this file), per the DSD-style architecture discussed.

Output format matches motion_planner.build_motion_plan's expected input
exactly: a list of strokes, each an ordered list of (x, y) points, pen
implicitly up between strokes. Units here are pixels (matching whatever
scale AuthorStyle.xHeightPx was measured in) -- converting px -> mm for the
real gantry is a later, separate calibration step (see motion_planner.py's
CONFIG), deliberately not conflated with style application here.
"""

import math

from base_alphabet import get_glyph_strokes


def style_text_to_strokes(text, style, fontName="futural", wordGapMultiplier=3.0):
    """text: any string (letters this author never wrote in training are
    fine -- that's the whole point). style: an AuthorStyle from
    style_extraction.py. Returns strokes in px, positioned left-to-right
    along a single baseline at y=0 (caller offsets to wherever on the page/
    gantry this line should sit)."""
    shear = math.tan(math.radians(style.slantDeg))
    scale = style.xHeightPx  # base_alphabet glyphs are cap-height=1.0, so multiplying by xHeightPx sizes them correctly
    letterGapPx = style.spacingRatio * style.xHeightPx
    wordGapPx = letterGapPx * wordGapMultiplier

    strokes = []
    cursorX = 0.0
    for ch in text:
        if ch == " ":
            cursorX += wordGapPx
            continue
        glyphStrokes, advance = get_glyph_strokes(ch, fontName)
        for stroke in glyphStrokes:
            styled = []
            for x, y in stroke:
                # shear first (pivoting around the glyph's own baseline, y=0
                # in base_alphabet's convention), then scale to this
                # author's real size, then place along the cursor
                shearedX = x + shear * y
                styled.append((shearedX * scale + cursorX, y * scale))
            strokes.append(styled)
        cursorX += advance * scale + letterGapPx

    return strokes


if __name__ == "__main__":
    import os
    import sys

    from author_style_cache import load_author_style, build_and_cache_author_style

    cacheDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN", "NOGIT", "AuthorStyles")
    authorId = input("Author id (blank = 'dylan' -- run author_style_cache.py first to create one): ").strip() or "dylan"
    text = input("Text to synthesize in that author's style (blank = 'testing new words'): ").strip() or "testing new words"

    try:
        style, embedding = load_author_style(authorId, cacheDir)
        print(f"Loaded cached style for '{authorId}' (no re-segmentation, no CNN forward pass): {style}")
    except FileNotFoundError:
        print(f"No cached style for '{authorId}' yet -- run author_style_cache.py first. "
              f"Falling back to extracting it fresh from baseline_model.png just for this demo.")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN"))
        import NonDatasetSegmenterFP as Segmenter
        from style_extraction import extract_author_style
        defaultImg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CNN",
                                   "NOGIT", "NonDatasetImages", "baseline_model.png")
        results, preview, meta = Segmenter.ProcessPage(defaultImg)
        style = extract_author_style(results, meta)

    strokes = style_text_to_strokes(text, style)
    print(f"Generated {len(strokes)} strokes for '{text}'")

    from rasterize import rasterize_strokes
    img = rasterize_strokes(strokes, lineWidthPx=max(1, int(style.xHeightPx * 0.06)))
    outPath = "styled_text_preview.png"
    img.save(outPath)
    print(f"Saved preview to {outPath}")
