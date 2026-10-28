# Backup

Files kept whole, as they were before a change large enough that reverting by
hand would be worse than copying the original back.

## segment_prepitchfix_2026-10-28.py

`segment.py` as it stood before the line-pitch fix of 28 October 2026.

`LinePitch` found peaks in the row-ink profile and treated every peak as a line
of text. On a page holding ONE line, the ascender band, the x-height band and
the descender band each raise their own peak, so the gaps between them were
read as line spacing: owen's short page reported a pitch of 15 against a true
51. Everything downstream is measured in that pitch, so `_BuildLineCrop` padded
by 2 px and threw away anything more than 12 px from the main run, which is
exactly where ascenders and descenders live. The crop handed to the reader held
44 per cent of the ink on the page and `this gantry writes today` came back as
`t7is (antrl writes tocal`.

Pages with many lines were never affected: real photographed pages give the
peak-gap method genuine line peaks and it measures them correctly.

To compare against the old behaviour without this file, set
`TRUST_ALL_PEAK_GAPS = True` in segment.py.
