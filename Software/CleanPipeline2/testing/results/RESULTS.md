# Handwriting robot: per writer results, every pipeline stage

Produced by `testing/full_results.py` on the existing trained weights; nothing was retrained.

Writers 150 to 588 come from the public database, the five named writers were photographed for this project.


## 1. Segmentation

Ground truth is the transcript typed when the page was collected, so the line count is exact. Photographed writers only.

| writer | pages | lines found | lines true | exact | s/page mean | s min | s max | pitch px | skew deg |
|---|---|---|---|---|---|---|---|---|---|
| abhinav | 8 | 206 | 206 | yes | 8.35 | 7.49 | 9.45 | 48-51 | -0.5..+1.5 |
| dylan | 5 | 124 | 124 | yes | 8.38 | 7.97 | 9.0 | 47-50 | +0.0..+1.0 |
| owen | 7 | 148 | 148 | yes | 7.67 | 6.12 | 8.78 | 50-58 | -1.5..+3.0 |
| thiya | 5 | 75 | 75 | yes | 5.73 | 4.52 | 6.77 | 49-53 | +0.0..+1.5 |
| yeukita | 5 | 116 | 116 | yes | 7.44 | 6.45 | 8.3 | 49-51 | -0.5..+1.5 |

## 2. Text recognition on real crops

Crops whose count matches the transcript, so each line pairs with its label. Measured with BOTH text checkpoints: scratch is the one the loader picks by default, joint is the one the report figures were taken on.

| writer | crops | char % scratch | word % scratch | char % joint | word % joint | ms/char |
|---|---|---|---|---|---|---|
| abhinav | 206 | 25.6 | 0.91 | 98.93 | 95.37 | 24.65 |
| dylan | 124 | 23.19 | 0.38 | 99.01 | 96.66 | 16.0 |
| owen | 148 | 23.56 | 0.06 | 99.14 | 96.47 | 19.29 |
| thiya | 75 | 25.4 | 0.12 | 96.49 | 85.42 | 18.31 |
| yeukita | 116 | 22.2 | 0.14 | 98.3 | 92.07 | 18.39 |

## 3. Writer identification, ink classifier

Raw grey crops, which is what the live server uses on a photograph.

| writer | crops | accuracy % |
|---|---|---|
| abhinav | 206 | 100.0 |
| dylan | 124 | 99.19 |
| owen | 148 | 99.32 |
| thiya | 75 | 100.0 |
| yeukita | 116 | 100.0 |

## 4. Writer identification, shape classifier

The same crops after stroke normalising, which is the judge used on drawn output.

| writer | crops | accuracy % |
|---|---|---|
| abhinav | 206 | 100.0 |
| dylan | 124 | 94.35 |
| owen | 148 | 99.32 |
| thiya | 75 | 94.67 |
| yeukita | 116 | 100.0 |

## 5. Style profiles

What the builder extracted per writer, and how far the library is trusted. lam is what AutoLegibility derives from the median prior distance.

| writer | chars | variants | prototypes | slant deg | x-height px | ascender | descender | word gap | connectedness | median priorD | lam | stored lam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 | 42 | 296 | 23 | -4.5 | 47.0 | 1.58 | -0.72 | 1.33 | 0.343 | 0.197 | 0.0 | - |
| 151 | 43 | 287 | 24 | 33.0 | 22.0 | 2.69 | -1.09 | 1.59 | 0.968 | 0.3073 | 0.841 | - |
| 152 | 38 | 273 | 24 | 19.5 | 46.0 | 1.48 | -0.79 | 1.47 | 0.715 | 0.2229 | 0.0 | - |
| 153 | 37 | 283 | 23 | 30.0 | 33.0 | 1.59 | -1.1 | 1.59 | 0.961 | 0.242 | 0.025 | - |
| 384 | 36 | 252 | 21 | 4.5 | 43.0 | 1.93 | -0.71 | 1.37 | 0.5 | 0.1962 | 0.0 | - |
| 551 | 38 | 271 | 25 | 42.0 | 20.0 | 1.84 | -0.67 | 2.68 | 0.557 | 0.2349 | 0.0 | - |
| 552 | 40 | 288 | 29 | -10.5 | 32.0 | 1.62 | -0.56 | 1.53 | 0.609 | 0.2377 | 0.0 | - |
| 588 | 41 | 254 | 25 | 0.0 | 29.0 | 1.73 | -0.68 | 2.07 | 0.668 | 0.24 | 0.0 | - |
| abhinav | 56 | 335 | 31 | 6.0 | 28.0 | 1.78 | -0.84 | 2.13 | 0.789 | 0.1797 | 0.0 | - |
| dylan | 42 | 302 | 26 | -6.0 | 26.0 | 1.68 | -0.62 | 1.79 | 0.195 | 0.1842 | 0.0 | - |
| owen | 43 | 307 | 25 | 0.0 | 32.0 | 1.51 | -0.68 | 1.1 | 0.319 | 0.1912 | 0.0 | - |
| thiya | 42 | 239 | 22 | 30.75 | 15.0 | 2.24 | -1.0 | 1.65 | 0.962 | 0.3177 | 0.971 | - |
| yeukita | 58 | 333 | 27 | 0.0 | 27.0 | 1.45 | -0.58 | 0.94 | 0.417 | 0.2237 | 0.0 | - |

## 6. Synthesis, judged by both networks

Best of 6, two sentences, one seed. Both judges also drive the search, so these figures are not independent of it.

| writer | char % | writer conf % | identified | s/line |
|---|---|---|---|---|
| 150 | 90.91 | 89.28 | yes | 9.4 |
| 151 | 68.18 | 67.38 | yes | 13.2 |
| 152 | 79.55 | 79.35 | yes | 10.7 |
| 153 | 70.45 | 56.03 | yes | 11.3 |
| 384 | 90.91 | 97.13 | yes | 10.0 |
| 551 | 61.36 | 73.49 | yes | 15.4 |
| 552 | 79.55 | 55.65 | yes | 10.3 |
| 588 | 75.0 | 63.65 | yes | 11.7 |
| abhinav | 86.36 | 99.77 | yes | 11.2 |
| dylan | 84.09 | 49.83 | no | 10.0 |
| owen | 97.73 | 99.35 | yes | 7.2 |
| thiya | 65.91 | 89.64 | yes | 12.2 |
| yeukita | 90.91 | 91.58 | yes | 9.1 |

## 7. G-code round trip

Geometry written out, then read back by a separate parser and compared. This measures rounding, stroke reordering and clamping only, not the machine.

| writer | IoU | mean err mm | max err mm |
|---|---|---|---|
| 150 | 0.9974 | 0.0004 | 0.0007 |
| 151 | 0.9985 | 0.0004 | 0.0007 |
| 152 | 0.9979 | 0.0004 | 0.0007 |
| 153 | 0.9986 | 0.0004 | 0.0007 |
| 384 | 0.9984 | 0.0004 | 0.0007 |
| 551 | 0.9984 | 0.0004 | 0.0007 |
| 552 | 0.9986 | 0.0004 | 0.0007 |
| 588 | 0.999 | 0.0004 | 0.0007 |
| abhinav | 0.9989 | 0.0004 | 0.0007 |
| dylan | 0.9974 | 0.0004 | 0.0007 |
| owen | 0.9976 | 0.0004 | 0.0007 |
| thiya | 0.9986 | 0.0004 | 0.0007 |
| yeukita | 0.9992 | 0.0004 | 0.0007 |
