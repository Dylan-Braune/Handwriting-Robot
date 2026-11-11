"""Copy the regenerated figures into the report revision.

Every figure named here is drawn by project code from project data. Each is
authored at exactly the report's text column, 455.24 pt = 6.30 in, and is
included at width=\\linewidth, so LaTeX does not rescale it and the type
sizes set in the figure are the sizes the reader gets.

Sources:
  Report/DesignVisualTests_2026-11-01/figures   the T* visual tests, and the
      two short Design versions D25 and D32, which the same functions draw
      from the same data as their full appendix counterparts
  Report/DesignResultsDiscussion_2026-11-01/Figures   the design-* subsystem
      illustrations and the results_* plots
  written in place by tmp/redraw_confusion.py   the two confusion matrices

Nothing here runs an experiment, trains, or touches hardware.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT/"Report/SubmissionRevision_2026-11-02/Figures"
VT = ROOT/"Report/DesignVisualTests_2026-11-01/figures"
DRD = ROOT/"Report/DesignResultsDiscussion_2026-11-01/Figures"

#the twelve visual tests the report includes, under their own names
VISUAL = ["T25_seam_cost_field","T27_ownership_corrections","T32_feature_stack",
          "T36_ctc_lattice","T38_gradient_check","T40_stroke_width_sweep",
          "T41_skeleton_trace_simplify","T43_glyph_coordinates",
          "T45_variant_pool_medoid","T49_connectedness",
          "T52_bresenham_staircase","T56_time_budget"]

#the two shortened Design figures, renamed to the names the LaTeX expects
SHORT = {"D25_seam_design":"design-seam-two-panels",
         "D32_first_features":"design-owen-first-features"}

#the subsystem illustrations and the results plots.  the six drawn by
#run_visual_bench.py are laid out on a 720 pt canvas, which LaTeX scales to
#the column by 455.24/720; their type is set that many times larger so it
#prints at the size asked for, so 10.00 in is the width to expect from them
BENCH = 720.0/72.0
DESIGN = [("design-segmentation-stages",BENCH),
          ("design-segmentation-variation",BENCH),
          ("design-stroke-normalisation",None),
          ("design-glyph-alignment",BENCH),
          ("design-candidate-selection",BENCH),
          ("design-style-gcode-examples",BENCH),
          ("design-physical-page-example",BENCH)]
RESULTS = ["results_writer_performance","results_physical_examples",
           "results_physical_style_scatter","results_r3_confirmed",
           "results_r4_stages","results_geometry_errors"]

#these two are written straight into the report by tmp/redraw_confusion.py,
#so they are only checked here, never copied over
INPLACE = ["results_reproduced_confusion","results_holdout_confusion"]

WANT = 6.30
missing,bad = [],[]


def Width(p):
    try:
        from pypdf import PdfReader
        return float(PdfReader(str(p)).pages[0].mediabox.width)/72.0
    except Exception:
        return None


def Copy(src,dst,want=None):
    if not src.exists():
        missing.append(str(src));return
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    w = Width(dst)
    if w is not None and abs(w-(WANT if want is None else want))>0.05:
        bad.append((dst.name,round(w,2)))
    print(f"  {dst.relative_to(REP.parent)}  {w:.2f} in" if w else f"  {dst.name}")


print("visual tests")
for s in VISUAL:
    Copy(VT/f"{s}.pdf",REP/"VisualTests"/f"{s}.pdf")
print("short Design versions")
for s,d in SHORT.items():
    Copy(VT/f"{s}.pdf",REP/f"{d}.pdf")
print("subsystem illustrations")
for s,want in DESIGN:
    Copy(DRD/f"{s}.pdf",REP/f"{s}.pdf",want)
print("results plots")
for s in RESULTS:
    Copy(DRD/f"{s}.pdf",REP/f"{s}.pdf")
print("already written in place")
for s in INPLACE:
    p = REP/f"{s}.pdf"
    w = Width(p)
    if w is None:
        missing.append(str(p))
    else:
        print(f"  {s}.pdf  {w:.2f} in")
        if abs(w-WANT)>0.05:
            bad.append((p.name,round(w,2)))

if missing:
    print("\nMISSING:");[print("  "+m) for m in missing]
if bad:
    print("\nNOT ONE COLUMN WIDE (will be rescaled by LaTeX):")
    [print(f"  {n}  {w} in") for n,w in bad]
sys.exit(1 if (missing or bad) else 0)
