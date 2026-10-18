"""Consolidate every measurement into one file, in consistent units
(all accuracies in PERCENT), each tagged with its provenance.

Nothing here is estimated: every value is read from a file a real run wrote.
"""
import json
import re
from pathlib import Path

# path setup for the clean package layout
import sys as _sys
from pathlib import Path as _p
_root = _p(__file__).resolve().parent.parent
_cnn = _root / 'Software' / 'CNN'
_ar = _cnn / 'AuthorReproductionStuff'
_gantry = _root / 'Software' / 'GantryControl'
for _q in (_cnn, _ar, _ar / 'np_inference', _gantry):
    if str(_q) not in _sys.path:
        _sys.path.insert(0, str(_q))


HERE = Path(__file__).resolve().parent
CNN = HERE.parents[1]
M = {}


def add(group, key, value, prov, unit=""):
    M.setdefault(group, {})[key] = dict(value=value, unit=unit, prov=prov)


def pct(x):
    if x is None:
        return None
    return round(x * 100, 2) if x <= 1.0001 else round(float(x), 2)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def readlog(p):
    """Read a log whatever wrote it.

    PowerShell redirection (``>`` / Out-File) produces UTF-16LE with a BOM,
    while Python's own prints produce UTF-8. Reading a UTF-16 log as UTF-8
    yields text full of NUL bytes, which silently defeats every regex -- this
    cost a debugging cycle, so detect the encoding here.
    """
    p = Path(p)
    if not p.exists():
        return ""
    b = p.read_bytes()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    if b[:3] == b"\xef\xbb\xbf":
        return b.decode("utf-8-sig")
    return b.decode("utf-8", errors="replace")


# ------------------------------------------------------------- text (R2)
t = (HERE / "stage3_text_accuracy.log")
if t.exists():
    s = readlog(t)
    m = re.search(r"OVERALL: char-acc ([\d.]+)% \| word-acc ([\d.]+)% \| (\d+) lines", s)
    if m:
        add("text_real", "char_pct", float(m.group(1)), "stage3_text_accuracy.log", "%")
        add("text_real", "word_pct", float(m.group(2)), "stage3_text_accuracy.log", "%")
        add("text_real", "n_lines", int(m.group(3)), "stage3_text_accuracy.log")
    m = re.search(r"TIMING: ([\d.]+)s total inference \| ([\d.]+) ms/line \| ([\d.]+) ms/char", s)
    if m:
        add("text_real", "ms_per_line", float(m.group(2)), "stage3_text_accuracy.log", "ms")
        add("text_real", "ms_per_char", float(m.group(3)), "stage3_text_accuracy.log", "ms")
    for a in ("Abhinav", "Dylan", "Owen", "Thiya", "Yeukita"):
        m = re.search(rf"{a}: char-acc ([\d.]+)% \| word-acc ([\d.]+)% \| (\d+) lines", s)
        if m:
            add("text_real_per_author", a.lower(),
                dict(char_pct=float(m.group(1)), word_pct=float(m.group(2)),
                     n=int(m.group(3))), "stage3_text_accuracy.log", "%")

# ------------------------------------------- writer-ID on real lines (R2)
s4 = jload(HERE / "stage4_summary.json")
if s4:
    add("writerid_real", "ink_overall_pct", pct(s4["overall"]["ink"]["acc"]),
        "stage4_summary.json, 1358 real lines", "%")
    add("writerid_real", "shape_overall_pct", pct(s4["overall"]["shape"]["acc"]),
        "stage4_summary.json, 1358 real lines", "%")
    add("writerid_real", "n_lines", s4["n_lines"], "stage4_summary.json")
    for model in ("ink", "shape"):
        for a, d in s4["per_author"][model].items():
            add(f"writerid_real_{model}", a,
                dict(acc_pct=pct(d["acc"]), n=d["total"], conf_pct=pct(d["mean_conf"])),
                f"stage4_summary.json ({model} head)", "%")
    for r in s4["agreement"]:
        add("agreement", r["model"],
            dict(top1_pct=pct(r["top1_agreement"]), max_abs_logit_diff=r["max_logit_diff"],
                 n=r["n"]), "numpy vs torch agreement", "%")
    add("writerid_real", "elapsed_s", s4.get("elapsed_seconds"),
        "stage4_summary.json", "s")

s4o = jload(HERE / "stage4_summary_preshape-fix.json")
if s4o:
    add("writerid_real_prefix", "shape_overall_pct", pct(s4o["overall"]["shape"]["acc"]),
        "stage4_summary_preshape-fix.json (before the training fix)", "%")
    for a, d in s4o["per_author"]["shape"].items():
        add("writerid_real_prefix_shape", a, dict(acc_pct=pct(d["acc"]), n=d["total"]),
            "before the classifier training fix", "%")

# --------------------------------------------------- end-to-end (R1, R3)
g2 = jload(HERE / "gcode_v2_summary.json")
if g2:
    add("end2end", "shape_render_pct", pct(g2["mean"]["shape_render"]),
        "gcode_v2_summary.json: 13 writers, 3 novel sentences, best-of-8", "%")
    add("end2end", "shape_gcode_pct", pct(g2["mean"]["shape_gcode"]),
        "gcode_v2_summary.json", "%")
    add("end2end", "char_render_pct", pct(g2["mean"]["char_render"]),
        "gcode_v2_summary.json", "%")
    add("end2end", "char_gcode_pct", pct(g2["mean"]["char_gcode"]),
        "gcode_v2_summary.json", "%")
    add("end2end", "word_gcode_pct", pct(g2["mean"]["word_gcode"]),
        "gcode_v2_summary.json", "%")
    add("end2end", "ink_gcode_pct", pct(g2["mean"]["ink_gcode"]),
        "gcode_v2_summary.json (ink head never trained on synthesis)", "%")
    for a, d in g2["per_author"].items():
        add("end2end_per_author", a,
            dict(shape_render_pct=pct(d["shape_render"]), shape_gcode_pct=pct(d["shape_gcode"]),
                 ink_gcode_pct=pct(d["ink_gcode"]), char_gcode_pct=pct(d["char_gcode"]),
                 word_gcode_pct=pct(d["word_gcode"])),
            "gcode_v2_summary.json", "%")
    add("end2end", "tries", g2["tries"], "gcode_v2_summary.json")
    add("end2end", "n_sentences", len(g2["sentences"]), "gcode_v2_summary.json")
    add("end2end", "elapsed_s", g2.get("elapsed_seconds"), "gcode_v2_summary.json", "s")

s5f = jload(HERE / "stage5_fast_summary.json")
if s5f:
    add("end2end_prefix", "shape_render_pct", pct(s5f["mean"]["idR"]),
        "stage5_fast_summary.json (before the training fix)", "%")
    add("end2end_prefix", "char_render_pct", pct(s5f["mean"]["charR"]),
        "stage5_fast_summary.json (before the fix)", "%")

# --------------------------------------------------- official rewrite (R1)
rl = CNN / "NOGIT" / "EndToEnd" / "rewrite_full.log"
if rl.exists():
    s = readlog(rl)
    m = re.search(r"writer-ID : ([\d.]+)%\s+\(through G-code ([\d.]+)%\)", s)
    if m:
        add("official_rewrite", "writerid_pct", float(m.group(1)),
            "NOGIT/EndToEnd/rewrite_full.log: 13 writers, 6 sentences x 2 seeds x N=30", "%")
        add("official_rewrite", "writerid_gcode_pct", float(m.group(2)),
            "rewrite_full.log", "%")
    m = re.search(r"text char : ([\d.]+)%\s+\(through G-code ([\d.]+)%\)", s)
    if m:
        add("official_rewrite", "char_pct", float(m.group(1)), "rewrite_full.log", "%")
        add("official_rewrite", "char_gcode_pct", float(m.group(2)), "rewrite_full.log", "%")
    m = re.search(r"text word : ([\d.]+)%\s+\(through G-code ([\d.]+)%\)", s)
    if m:
        add("official_rewrite", "word_pct", float(m.group(1)), "rewrite_full.log", "%")
    m = re.search(r"REAL handwriting : char ([\d.]+)%\s+word ([\d.]+)%\s+\((\d+) lines\)", s)
    if m:
        add("matched_reference", "real_char_pct", float(m.group(1)), "rewrite_full.log", "%")
        add("matched_reference", "real_word_pct", float(m.group(2)), "rewrite_full.log", "%")
        add("matched_reference", "n_lines", int(m.group(3)), "rewrite_full.log")
    m = re.search(r"SYNTHESIZED\s+: char ([\d.]+)%\s+word ([\d.]+)%", s)
    if m:
        add("matched_reference", "synth_char_pct", float(m.group(1)), "rewrite_full.log", "%")
        add("matched_reference", "synth_word_pct", float(m.group(2)), "rewrite_full.log", "%")
    for m in re.finditer(r"^\s+(\w+): writer-ID\s+([\d.]+)% \(gcode\s+([\d.]+)%\) "
                         r"\| text char\s+([\d.]+)%", s, re.M):
        add("official_rewrite_per_author", m.group(1),
            dict(idR_pct=float(m.group(2)), idG_pct=float(m.group(3)),
                 charR_pct=float(m.group(4))), "rewrite_full.log", "%")

# ------------------------------------------------------- server / speed (R3,R4,R6)
s6 = jload(HERE / "stage6_results.json")
if s6:
    add("server", "n_authors", s6["authors"]["n"],
        "stage6_results.json (Flask test client, in-process)")
    add("server", "torch_imported", s6["import_closure"]["torch"],
        "stage6_results.json: torch absent from the boot closure")
    add("server", "authors_list_ms", s6["authors"].get("ms"), "stage6_results.json", "ms")
    cl = s6.get("classify") or {}
    r = cl.get("response") or {}
    if cl:
        add("server_classify", "latency_ms", cl.get("ms"), "stage6_results.json", "ms")
        add("server_classify", "predicted", r.get("predicted_author"), "stage6_results.json")
        add("server_classify", "confidence_pct", r.get("author_confidence_pct"),
            "stage6_results.json", "%")
        add("server_classify", "lines_detected", r.get("lines_detected"),
            "stage6_results.json")
    gn = s6.get("generate") or {}
    r = gn.get("response") or {}
    if gn:
        add("server_generate", "seconds", gn.get("seconds"),
            "stage6_results.json: /api/generate, 4-character word, N=6", "s")
        add("server_generate", "requested", gn.get("requested"), "stage6_results.json")
        add("server_generate", "read_back", r.get("read_back_text"), "stage6_results.json")
        add("server_generate", "text_accuracy_pct", r.get("text_accuracy_pct"),
            "stage6_results.json", "%")

# ------------------------------------------------------- stroke / pen (R1)
for name, grp in (("normalised_stroke.json", "stroke_measurement"),
                  ("thin_skeleton_results.json", "thin_skeleton"),
                  ("final_thickness_test.json", "thickness_identity"),
                  ("render_pen.json", "drawn_pen"),
                  ("gantry_written_classify.json", "robot_written")):
    d = jload(HERE / name)
    if d:
        add(grp, name.replace(".json", ""), d, f"{name} (this session)")

# ------------------------------------------------------------ inventory
inv = jload(CNN / "NOGIT" / "inventory" / "summary_v2.json")
if inv:
    add("code", "files", inv["n_files"], "inventory/summary_v2.json")
    add("code", "lines", inv["total_lines"], "inventory/summary_v2.json")
    add("code", "definitions", inv["total_defs"], "inventory/summary_v2.json")
    add("code", "dead_definitions", len(inv["unused"]), "inventory/summary_v2.json")
boot = jload(CNN / "NOGIT" / "inventory" / "boot_closure.json")
if boot:
    add("code", "boot_files", len(boot["boot"]), "inventory/boot_closure.json")
    add("code", "boot_lines", boot["boot_lines"], "inventory/boot_closure.json")

(HERE / "measurements.json").write_text(json.dumps(M, indent=2), encoding="utf-8")
print("wrote measurements.json\n")
for g in sorted(M):
    print(f"  {g:<30} {len(M[g])} values")

print("\n--- headline values (percent unless noted) ---")
for g, k in (("text_real", "char_pct"), ("text_real", "word_pct"),
             ("text_real", "ms_per_char"), ("writerid_real", "ink_overall_pct"),
             ("writerid_real", "shape_overall_pct"),
             ("writerid_real_prefix", "shape_overall_pct"),
             ("end2end", "shape_render_pct"), ("end2end", "char_render_pct"),
             ("end2end", "char_gcode_pct"),
             ("official_rewrite", "writerid_pct"), ("official_rewrite", "char_pct"),
             ("matched_reference", "real_char_pct"), ("matched_reference", "synth_char_pct"),
             ("server", "n_authors"), ("server_generate", "seconds"),
             ("server_classify", "latency_ms")):
    d = M.get(g, {}).get(k)
    print(f"  {g}.{k:<26} = {d['value'] if d else None} {d['unit'] if d else ''}")
