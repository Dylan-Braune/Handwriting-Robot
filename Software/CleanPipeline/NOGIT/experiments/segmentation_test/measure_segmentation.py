"""Measure the deployed segmenter against hand-written transcriptions.

Fills the two outstanding measurements of report Section 3.4:
  (a) detected line count against the hand-counted line count, per page;
  (b) wall-clock time of SegmentLean.SegmentLines, per page.

Ground truth is the number of non-empty lines in each page's label file,
which is the transcription typed by hand when the page was collected.

Writes new_results/segmentation_results.json and prints a table.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent.parent.parent          # Software/CleanPipeline
sys.path.insert(0, str(PIPE))

import AuthorsConfig as AC                                    # noqa: E402
import SegmentLean                                            # noqa: E402

OUT_DIR = HERE / "new_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def TruthLineCount(labelPath):
    """Non-empty transcription lines = the hand-counted line count."""
    if not labelPath.exists():
        return None
    text = labelPath.read_text(encoding="utf-8", errors="replace")
    return sum(1 for ln in text.splitlines() if ln.strip())


def main():
    rows = []
    for author in AC.PERSONAL_AUTHORS:
        for imgPath, labelPath in AC.PersonalAuthorPages(author):
            truth = TruthLineCount(labelPath)
            if truth is None:
                print("skip (no label): %s" % imgPath.name)
                continue
            t0 = time.time()
            try:
                lines, _ov, stats = SegmentLean.SegmentLines(str(imgPath))
            except Exception as exc:
                print("FAIL %-28s %s: %s" % (imgPath.name, type(exc).__name__, exc))
                rows.append(dict(author=author, page=imgPath.name, truth=truth,
                                 detected=None, error="%s: %s" % (type(exc).__name__, exc)))
                continue
            wall = time.time() - t0
            rows.append(dict(author=author, page=imgPath.name, truth=truth,
                             detected=int(stats["nLines"]),
                             seconds=float(stats["seconds"]),
                             wall=round(wall, 2),
                             pitch=int(stats["pitch"]),
                             angle=float(stats["angle"]),
                             paperCoverage=float(stats["paperCoverage"]),
                             removedStraightPixels=int(stats["removedStraightPixels"]),
                             artifactPieces=int(stats["artifactPieces"])))
            print("%-10s %-24s truth=%3d detected=%3d  %+d  %6.1f s  P=%d  ang=%+.1f"
                  % (author, imgPath.name, truth, stats["nLines"],
                     stats["nLines"] - truth, stats["seconds"],
                     stats["pitch"], stats["angle"]))

    ok = [r for r in rows if r.get("detected") is not None]
    exact = [r for r in ok if r["detected"] == r["truth"]]
    print()
    print("pages attempted           : %d" % len(rows))
    print("pages segmented           : %d" % len(ok))
    print("exact line-count match    : %d of %d (%.1f%%)"
          % (len(exact), len(ok), 100.0 * len(exact) / max(1, len(ok))))
    if ok:
        diffs = [r["detected"] - r["truth"] for r in ok]
        secs = [r["seconds"] for r in ok]
        totT = sum(r["truth"] for r in ok)
        totD = sum(r["detected"] for r in ok)
        print("line-count difference     : min %+d, max %+d" % (min(diffs), max(diffs)))
        print("total lines               : %d detected against %d hand-counted (%.1f%%)"
              % (totD, totT, 100.0 * totD / max(1, totT)))
        print("per-page time             : %.1f to %.1f s, mean %.1f s"
              % (min(secs), max(secs), sum(secs) / len(secs)))

    (OUT_DIR / "segmentation_results.json").write_text(
        json.dumps(dict(rows=rows), indent=1), encoding="utf-8")
    print("\nwrote %s" % (OUT_DIR / "segmentation_results.json"))


if __name__ == "__main__":
    main()
