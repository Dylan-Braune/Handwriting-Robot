# Captions and provenance

The caption belongs below the figure in the report, set in bold, as
Appendix 4 requires; it is kept here rather than drawn onto the
picture.  The source line records which code and which data made the
figure.


## T41_skeleton_trace_simplify

T41 How an area of ink becomes a pen path. Thinning gives a one-pixel centreline, the centreline is traced as a graph split at junctions with short spurs pruned and near-collinear ends rejoined, and the result is simplified within a bounded deviation. The shading separates the traced paths: where one letter is drawn in two shades the tracer has chosen a pen lift the writer may never have made, because a static image cannot record stroke order.

Source: `segment.Skeletonize / style.TracePolylines / SimplifyPolyline | Owen01 line 00`

## T42_cut_cost_refinement

T42 Refining a character boundary. The recogniser locates a character to a frame span, whose midpoint is only approximately where the letters separate. The cost prefers a column that is thin and low, which in cursive is the ligature crossing, and a genuinely blank column is given zero cost outright. One shared increasing boundary array is enforced so two neighbouring cuts cannot land on the same gap and leave an empty cell.

Source: `style.ExtractLineGlyphs cut cost | Owen01 line 00`

## T43_glyph_coordinates

T43 The transform that makes a glyph reusable. Image rows increase downwards and a letter's size depends on the photograph; the stored frame measures height upwards from the baseline in x-heights and removes the writer's slant, which is reapplied at placement. The lead is stored separately from the path, so the writer's own gap between the cell edge and the first ink survives instead of every letter being butted to its cell. Illustration of Equation 3.16.

Source: `style.ExtractLineGlyphs | Owen01 line 00`

## T45_variant_pool_medoid

T45 Keeping variation without keeping extraction errors. Every stored instance of one letter is overlaid in silver; the variant with the smallest total descriptor distance to the others is the medoid, an actual example the writer produced. The prototype beside it is a robust summary of the compatible variants and is a different object: one selects, the other combines. Shown for three writers, since the spread of a pool is a property of the hand and of how cleanly it segmented.

Source: `style._ShapeGrid, final stored profiles`

## T46_shape_descriptor

T46 The descriptor the shape prior and the consensus filter operate on. A glyph's strokes are reduced to a smoothed, normalised occupancy grid, so two forms are compared by where the ink falls rather than by how many points a path happens to contain. The same letter from different writers gives visibly different grids, which is what makes the comparison informative within a writer and tolerant across writers.

Source: `style._ShapeGrid`

## T47_slant_search

T47 Slant estimated over a whole line rather than from one letter. The ink is sheared through candidate angles and the angle giving the most concentrated column profile is stored. Different writers put the peak in clearly different places, which is what makes slant a usable style quantity; a flat curve would mean the measure could not separate them. It remains an estimate affected by letter content, not an observation of the hand.

Source: `segment.EstimateSlantDeg`

## T49_connectedness

T49 Connectedness as the implementation defines it: the fraction of eligible adjacent non-space glyph boundaries that a traced centreline actually crosses, here 3/39 = 0.0769 on one real line. It is measured on the writer's own ink, not by counting pen lifts in a generated file, which would instead measure the extractor's and the synthesiser's decisions. Illustration of Equation 3.18.

Source: `style.ExtractLineGlyphs | Owen01 line 00`

## T19_glyph_quality_gates

T19 The variants that survived the library gates, with the two quantities the gates act on printed on each. Width is capped at 2.1 x-heights for ordinary characters and 2.9 for wide ones, and the height class is checked against what the letter should reach, so an 'l' that never rises above the band is rejected as a mis-cut rather than stored as a narrow letter. These are the pools after filtering: the gates only ever reject, never invent.

Source: `style gates, owen_curated.json`
