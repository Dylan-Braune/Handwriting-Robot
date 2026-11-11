# Quick cursive and connected-word test, 2 November 2026

The geometric comparison can be calculated on connected words without splitting characters. In this small test it produces strong matches for some partly connected Abhinav words, but does not establish strong overall matching for Thiya's fully cursive output.

| Writer | Words | Prior line-x-height rule, mean F | Separate word-height normalisation, mean F | Best word-height example |
|---|---:|---:|---:|---|
| Thiya | 8 | 18.79% | 55.18% | but, 71.14% |
| Abhinav | 17 | 44.26% | 67.38% | of, 96.16% |

Five of Abhinav's 17 words exceed F = 0.85 under word-height normalisation. None of Thiya's eight do. No word exceeds 0.85 under the unchanged line-x-height rule. The word-height result is a separate diagnostic, not a replacement for the previous character experiment.

## What was done

Used the existing physical source-text word pairs, eight Thiya and seventeen Abhinav, omitting Abhinav's single-letter a because this test concerns words. No per-character cuts or recognition models were used. Original same-word banks were selected from the existing personal dataset by page diversity and source order, with up to eight references per word. Exact transcript tokens were retained, including case and punctuation. There were 201 candidate reference entries from 112 relevant source lines plus earlier fixed references. One hundred bank entries were selected before scoring. Two with obvious foreign ink were excluded after visual review, before scores. Some entries are different cached representations of the same source occurrence, so bank counts are not counts of independent handwritten words.

The primary calculation preserves the previous 48-pixel line x-height, isotropic scaling, bounded translation of five pixels and nearest-centreline F-score within 4.8 pixels. Each physical word takes its best score within its fixed original bank. There is no width stretching, slant correction or deformable alignment. Because whole words retain spacing and join geometry, these word averages are not directly comparable with the earlier per-character means.

The separate diagnostic tightly crops each word and scales its own total ink height to 48 pixels, preserving its aspect ratio. It uses the same five-pixel translation limit and a 4.8-pixel distance tolerance, now relative to total word height rather than line x-height. It independently chooses the best reference from the same fixed bank. It removes absolute word-height differences and changes the physical meaning of the tolerance, so the two columns must remain separately labelled.

Whole-word overlays show differences in relative size, character spacing and joins. These are possible contributors to low overlap. This test does not isolate the contributions of normalisation, glyph extraction, synthesis or gantry motion. Letter splitting in fully connected writing has not been validated, so no cursive per-character accuracy is claimed.

## Saved evidence

figures/selected_connected_examples.png and its PDF show Abhinav of and the, and Thiya but. Their diagnostic scores are 96.16%, 86.47% and 71.14%. These examples were selected after scoring. Per-writer connected_words images show strongest, middle-ranked and weakest primary cases. All 25 word results and all bank comparisons are retained in results.json. word_scores.csv provides the concise numerical table.

quick_cursive_test.py regenerates scores and per-writer figures from inputs/. finalise_cursive.py regenerates the selected gallery, these notes and the evidence ZIP. The frozen source/segment.py supplies the same image toolkit. The preparation script requires the original repository and caches, while numerical regeneration uses the saved masks. Dependencies are numpy, scipy, pillow, opencv-python and matplotlib. The point-distance F definition is the same as in the previous study, based on the Tanks and Temples tutorial Appendix B, https://tanksandtemples.org/tutorial/. The word-height diagnostic is an exploratory project choice.

No report LaTeX, application code, model weights or hardware was changed. The earlier five-writer study and its package remain unchanged. This is an offline exploratory check, not an ODROID measurement or proof of the proposal requirements.
