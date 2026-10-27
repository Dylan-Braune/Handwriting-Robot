# Results

One folder per thing being tested. Each holds the images, the G-code and the
script that produced them, so a figure can be regenerated rather than trusted.

## ligature_joins/

The join between two letters, three ways, on `"remove the inconsistencies on
the belt pulley"` at nTries 5, seed 23, for the five photographed writers.

`sheet.png` is the full contact sheet (7400 x 1749): one row per writer, and
the columns are the original midpoint dip, the C1 join that ships, C1 plus the
tangent join, and the same trajectory taken back out of its G-code.
`sheet_top.png` and `sheet_bot.png` are the same thing split for viewing.

| writer | before C1 | C1 (shipped) | C1 + tangent |
|---|---|---|---|
| abhinav | 48.9 / 99.7 | 46.7 / 99.6 | 42.2 / 99.9 |
| dylan | 80.0 / 90.2 | 80.0 / 91.1 | 82.2 / 83.3 |
| owen | 97.8 / 100 | 100 / 100 | 100 / 100 |
| thiya | 31.1 / 95.2 | 26.7 / 94.8 | 31.1 / 97.0 |
| yeukita | 97.8 / 45.0 | 97.8 / 44.7 | 97.8 / 49.0 |

*(character accuracy / writer confidence, per cent)*

Read with care: this sentence is 44 characters where the published figures
were measured on 21, and nTries was 5 against the tuned 14. On the published
sentence at nTries 14 the same writers give abhinav 90.9 per cent and thiya
77.3, both above what `RESULTS.md` records. Sentence length moves these
figures by about 40 points, far more than any join does.

## gcode_path/

How the pen works its way over a retraced page, from `Dylan01.jpg`.

`path_rowtol_strokeheight.png` and `path_rowtol_linepitch.png` plot the
G-code: colour is draw order from blue to yellow, and the pink lines are
pen-up travel. `d01.gcode` and `d02.gcode` are the files themselves,
`plotgc.py` draws them.

Ordering groups strokes into rows by their height on the page. Taking the row
tolerance from stroke heights gives 2.04 mm, but a text line here is 5.82 mm
tall, so each line was treated as three rows. Passing the real line pitch
instead cuts pen-up travel from 4218 mm to 3754 mm, an 11 per cent saving.

The plots also show two faults in the retrace: the page is drawn bottom to
top, and mirrored vertically. Both come from negating the row coordinate to
convert page-down into millimetres-up, where it should be offset against the
page height.

## page_retrace/

`dylan01_traced_preview.png` is a photographed page traced straight into pen
paths, with no recognition and no synthesis: 24 lines, 1383 strokes, 42238
G-code lines. Drawing this and comparing it with the original sheet measures
the gantry on its own, since nothing in between made a decision that could
have changed the shapes.
