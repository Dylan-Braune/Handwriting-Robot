"""
base_alphabet.py

The "existing alphabet set" you described: a neutral, generic single-stroke
skeleton per character that per-author style gets applied ON TOP of, instead
of trying to learn stroke geometry from scratch per author (which your
photographed-page-only data can't support well -- see style_extraction.py's
docstring for why).

Uses the Hershey vector fonts (Hershey, "Calligraphy for Computers", US Naval
Weapons Laboratory, 1967) via the `HersheyFonts` PyPI package. Hershey fonts
were designed FOR pen plotters: each glyph is already a small set of strokes
(ordered points, pen lifts between strokes) rather than a filled outline, so
there's no separate "turn a font into a pen path" step to build -- that's
exactly the format motion_planner.py already expects (list of strokes, each
an ordered list of (x, y) points, pen up between strokes).

Install: pip install HersheyFonts

Units: Hershey's native unit has cap-height (cap_line to base_line) = 21
units for every default font. This module rescales that to cap-height = 1.0
("em-relative" units) so a style deformation can multiply by an author's
REAL measured x-height (in px or mm, whichever the caller is working in)
without caring about Hershey's internal numbering. Y is left in its native
sense (increasing downward, baseline near 0, ascenders negative, descenders
positive) to match the rest of this repo's image-coordinate convention
(NonDatasetSegmenterFP's bboxes are also top-down y).
"""

from HersheyFonts import HersheyFonts

_FONT_CACHE = {}


def _get_font(fontName="futural"):
    if fontName not in _FONT_CACHE:
        hf = HersheyFonts()
        hf.load_default_font(fontName)
        _FONT_CACHE[fontName] = hf
    return _FONT_CACHE[fontName]


def get_glyph_strokes(ch, fontName="futural"):
    """Returns (strokes, advance) for one character, in cap-height=1.0 units.
    strokes: list of strokes, each a list of (x, y) float tuples.
    advance: how far the "pen carriage" should move right before the next
    character (still in cap-height=1.0 units) -- Hershey's own per-glyph
    width, not yet touched by any author spacing style."""
    hf = _get_font(fontName)
    glyphs = list(hf.glyphs_for_text(ch))
    if not glyphs:
        return [], 0.6  # unknown glyph (e.g. unsupported punctuation) -- blank with a plausible advance

    glyph = glyphs[0]
    capHeight = glyph.base_line - glyph.cap_line  # constant per font (21 for every Hershey default font)
    scale = 1.0 / capHeight

    strokes = [[(x * scale, y * scale) for (x, y) in stroke] for stroke in glyph.strokes]
    advance = glyph.char_width * scale
    return strokes, advance


def text_to_base_strokes(text, fontName="futural"):
    """Lays out a whole string using ONLY Hershey's own advance widths (no
    author style yet) -- useful as a quick visual sanity check that the
    alphabet wrapper itself is correct before any style deformation is
    layered on top (see glyph_styler.py for the styled version, which is
    what you actually want to feed downstream)."""
    strokes = []
    cursorX = 0.0
    for ch in text:
        if ch == " ":
            cursorX += 0.6  # space width guess; real styling should use the author's measured word-gap instead
            continue
        glyphStrokes, advance = get_glyph_strokes(ch, fontName)
        for stroke in glyphStrokes:
            strokes.append([(x + cursorX, y) for (x, y) in stroke])
        cursorX += advance
    return strokes


if __name__ == "__main__":
    print("Available Hershey fonts:", HersheyFonts().default_font_names)
    fontChoice = input("Font to preview (blank = futural, or try 'cursive'): ").strip() or "futural"
    text = input("Text to lay out (blank = 'have a nice day'): ").strip() or "have a nice day"

    strokes = text_to_base_strokes(text, fontChoice)
    print(f"Generated {len(strokes)} strokes for '{text}' using font '{fontChoice}'")

    from PIL import Image, ImageDraw
    scale = 100
    pad = 20
    allX = [x for s in strokes for x, y in s]
    allY = [y for s in strokes for x, y in s]
    w = int((max(allX) - min(allX)) * scale) + 2 * pad
    h = int((max(allY) - min(allY)) * scale) + 2 * pad
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    minX, minY = min(allX), min(allY)
    for stroke in strokes:
        pts = [((x - minX) * scale + pad, (y - minY) * scale + pad) for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=2)
    outPath = "base_alphabet_preview.png"
    img.save(outPath)
    print(f"Saved preview to {outPath}")
