#score the fourteen reproduction scans against what the gantry was asked to draw.
#the block order is not the same on every sheet and the third block differs per
#writer, so accuracy is taken flat over the whole page by edit distance rather
#than line against line, which would punish a page merely for a different order.
import sys, json
from pathlib import Path
CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3))
import recognise as REC

OUT = Path(__file__).resolve().parent / "batch14"

PARA = ("This handwriting robot was designed to identify the words and writer of any "
        "piece of handwritten text, then reproduce new sentences in that same persons "
        "style using a gantry and pen. Over 13 different writers were used to train and "
        "test the system, each contributing several pages so the network could learn how "
        "they loop their letters, slant their lines and space their words.")
FOX = "The quick brown fox jumps over the lazy dog"

#block three, transcribed from each sheet by eye
B3 = {
 "iam_play": "This is not a filmed play. It has been conceived throughout in terms of the cinema",
 "384": "The equipment takes two forms - an insulating guard on the jib of the crane",
 "552": "come come to the true world. you become a true human.",
 "588": ("We didn't approach unchallenged, just short of the diggings there was a "
         "roughly-built shack"),
 "abhinav": ("Objects that travel at constant velocity and constant acceleration in a 2D "
             "space. It calculates the true values of the objects."),
 "dylan": "image is then converted to a binary ink mask using adaptive thresholding",
 "owen": ("Mainstream TV oftens involves contract and the need to pay for a subscription, "
          "while Youtube offers a free, easily accessible hobby."),
 "thiya": "Lithium is current standard but limited in capacity and sustainability.",
 "yeukita": ("in my setup, both filters are fed the exact same simulated measurement "
             "sequence in each run"),
}

#what each sheet actually carries, in the order it was drawn
TRUTH = {
 "Author150":  [PARA, FOX, B3["iam_play"]],
 "Author152":  [PARA, FOX, B3["iam_play"]],
 "Author1531": [PARA, FOX],
 "Author1532": [B3["iam_play"]],
 "Author3841": [PARA, FOX],
 "Author3842": [B3["384"]],
 "Author552":  [PARA, FOX, FOX, B3["552"]],
 "Author588":  [FOX, B3["588"], PARA],
 "AuthorA1":   [PARA],
 "AuthorA2":   [FOX, B3["abhinav"]],
 "AuthorD":    [PARA, FOX, B3["dylan"]],
 "AuthorO":    [PARA, FOX, B3["owen"]],
 "AuthorT":    [PARA, FOX, B3["thiya"]],
 "AuthorY":    [PARA, FOX, B3["yeukita"]],
}

summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
byfile = {r["file"]: r for r in summary}

print("%-12s %-8s %-8s %6s  %5s %5s  %5s %5s  %s" %
      ("page", "truth", "named", "conf", "lnDet", "lnExp", "char%", "word%", "chars"))
print("-" * 86)
rows = []
for stem, blocks in TRUTH.items():
    d = json.loads((OUT / (stem + ".json")).read_text(encoding="utf-8"))
    want = " ".join(blocks)
    got = " ".join(d["predicted_lines"])
    ca = 100 * REC.CharAcc(got, want)
    wa = 100 * REC.WordAcc(got, want)
    r = byfile[stem + ".jpg"]
    rows.append((stem, r, ca, wa, want, got, d))
    print("%-12s %-8s %-8s %5.1f%%  %5d %5s  %5.1f %5.1f  %d/%d" %
          (stem, r["truth"], r["pred"], r["conf"],
           d["lines_detected"], "?", ca, wa, len(got), len(want)))

print()
ok = sum(1 for _, r, *_ in rows if r["ok"])
print("writer identification: %d/%d = %.1f%%" % (ok, len(rows), 100.0 * ok / len(rows)))

#the dense pages are the ones carrying the full paragraph; the sparse ones are
#the single block sheets, which is where the line count ran away
dense = [t for t in rows if PARA in " ".join(TRUTH[t[0]])]
sparse = [t for t in rows if PARA not in " ".join(TRUTH[t[0]])]
for name, grp in (("pages carrying the paragraph", dense), ("single block pages", sparse)):
    if not grp:
        continue
    o = sum(1 for _, r, *_ in grp if r["ok"])
    print("  %-30s writer %d/%d   char %.1f%%   word %.1f%%" %
          (name, o, len(grp),
           sum(t[2] for t in grp) / len(grp), sum(t[3] for t in grp) / len(grp)))

print("\nmean char accuracy over all 14: %.1f%%" % (sum(t[2] for t in rows) / len(rows)))
print("mean word accuracy over all 14: %.1f%%" % (sum(t[3] for t in rows) / len(rows)))

print("\n\n=== spurious line check (detected vs ink actually on the sheet) ===")
for stem, r, ca, wa, want, got, d in rows:
    empties = sum(1 for l in d["lines"] if len(l["text_read"].strip()) <= 2)
    print("%-12s detected %2d  near-empty crops %2d  read %4d chars for %4d expected"
          % (stem, d["lines_detected"], empties, len(got), len(want)))

(OUT / "scored.json").write_text(json.dumps(
    [{"page": s, "truth": r["truth"], "named": r["pred"], "ok": r["ok"],
      "conf": r["conf"], "char_pct": round(ca, 1), "word_pct": round(wa, 1),
      "lines": r["lines"], "read": got, "want": want}
     for s, r, ca, wa, want, got, d in rows], indent=1), encoding="utf-8")
