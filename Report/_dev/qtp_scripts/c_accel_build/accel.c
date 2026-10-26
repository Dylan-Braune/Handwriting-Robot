/*
 * accel.c -- portable C99 acceleration kernels for SegmentPage.py.
 *
 * Implements (see README.md for the full story):
 *   - label_components : two-pass union-find connected-component labeling
 *                         on a flat uint8_t binary mask (0/1).
 *   - box_sum           : box-sum (not mean) over a (2*ry+1)x(2*rx+1)
 *                         window per pixel, with edge clamping, matching
 *                         RawImageOps.BoxSum's summed-area-table behaviour
 *                         exactly (including how truncated windows at
 *                         borders behave: the window just gets smaller,
 *                         the divisor is NOT renormalized by box_sum
 *                         itself -- that matches BoxSum, whose caller
 *                         BoxMean divides by BoxCount separately).
 *
 * Strict C99. No platform-specific intrinsics, no SIMD, no non-standard
 * headers. Must compile unmodified with:
 *   gcc -O3 -shared -fPIC -o libaccel.so accel.c -lm      (Linux / ODROID)
 *   gcc -O3 -shared -o libaccel.dll accel.c -lm           (Windows/MinGW)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#define ACCEL_EXPORT __declspec(dllexport)
#else
#define ACCEL_EXPORT
#endif

/* ===========================================================================
 * box_sum
 * ===========================================================================
 * img, out : flat row-major arrays of length h*w (double)
 * ry, rx   : window half-sizes; full window is (2*ry+1) x (2*rx+1)
 *
 * Builds a (h+1)x(w+1) summed-area table (double, zero row/col on top/left,
 * exactly like RawImageOps._Integral), then for every output pixel sums
 * the clamped window via four integral-table lookups -- identical maths to
 * the numpy version, just without the huge fancy-indexing temporaries.
 */
ACCEL_EXPORT void box_sum(const double *img, int h, int w, int ry, int rx,
                           double *out) {
    int64_t H = (int64_t)h + 1;
    int64_t W = (int64_t)w + 1;
    double *ii = (double *)calloc((size_t)(H * W), sizeof(double));
    if (ii == NULL) {
        return; /* allocation failure: leave out[] untouched */
    }

    /* row-wise cumulative sum, then column-wise cumulative sum, matching
     * np.cumsum(np.cumsum(img, axis=0), axis=1) written into ii[1:,1:] */
    for (int y = 0; y < h; y++) {
        double rowAcc = 0.0;
        const double *imgRow = img + (int64_t)y * w;
        double *iiRow = ii + (int64_t)(y + 1) * W;
        double *iiPrevRow = ii + (int64_t)y * W;
        for (int x = 0; x < w; x++) {
            rowAcc += imgRow[x];
            iiRow[x + 1] = iiPrevRow[x + 1] + rowAcc;
        }
    }

    for (int y = 0; y < h; y++) {
        int y1 = y - ry;
        int y2 = y + ry + 1;
        if (y1 < 0) y1 = 0;
        if (y1 > h) y1 = h;
        if (y2 < 0) y2 = 0;
        if (y2 > h) y2 = h;
        const double *iiY2 = ii + (int64_t)y2 * W;
        const double *iiY1 = ii + (int64_t)y1 * W;
        double *outRow = out + (int64_t)y * w;
        for (int x = 0; x < w; x++) {
            int x1 = x - rx;
            int x2 = x + rx + 1;
            if (x1 < 0) x1 = 0;
            if (x1 > w) x1 = w;
            if (x2 < 0) x2 = 0;
            if (x2 > w) x2 = w;
            outRow[x] = iiY2[x2] - iiY1[x2] - iiY2[x1] + iiY1[x1];
        }
    }

    free(ii);
}

/* ===========================================================================
 * label_components
 * ===========================================================================
 * Standard raster-scan two-pass union-find connected component labeling.
 * mask        : flat row-major uint8_t array of length h*w, values 0/1
 * connectivity: 4 or 8
 * out_labels  : flat row-major int32_t array of length h*w, written with
 *               0 for background and compact labels 1..N for foreground
 *
 * Returns N (the component count). Does NOT attempt to match the Python
 * run-based union-find's label NUMBERING (which depends on row-run scan
 * order) -- only the PARTITION of foreground pixels into components, and
 * the count, are guaranteed to match.
 */
static int32_t uf_find(int32_t *parent, int32_t a) {
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

static void uf_union(int32_t *parent, int32_t a, int32_t b) {
    int32_t ra = uf_find(parent, a);
    int32_t rb = uf_find(parent, b);
    if (ra != rb) {
        /* attach the numerically larger root under the smaller -- arbitrary
         * but deterministic, keeps find() shallow on average */
        if (ra < rb) {
            parent[rb] = ra;
        } else {
            parent[ra] = rb;
        }
    }
}

ACCEL_EXPORT int32_t label_components(const uint8_t *mask, int h, int w,
                                       int connectivity, int32_t *out_labels) {
    int64_t n = (int64_t)h * (int64_t)w;
    if (n == 0) {
        return 0;
    }

    /* provisional per-pixel labels (0 = unlabeled/background), plus a
     * union-find parent array sized for the worst case (every foreground
     * pixel its own label). Labels are 1-based so 0 can mean background. */
    int32_t *prov = (int32_t *)calloc((size_t)n, sizeof(int32_t));
    int32_t *parent = (int32_t *)malloc((size_t)(n + 1) * sizeof(int32_t));
    if (prov == NULL || parent == NULL) {
        free(prov);
        free(parent);
        return -1;
    }
    for (int64_t i = 0; i <= n; i++) {
        parent[i] = (int32_t)i;
    }

    int32_t nextProv = 1;
    int eightConn = (connectivity == 8);

    /* pass 1: raster scan, look at already-visited neighbours
     * (west, north, and for 8-connectivity north-west, north-east) */
    for (int y = 0; y < h; y++) {
        const uint8_t *row = mask + (int64_t)y * w;
        const uint8_t *prevRow = (y > 0) ? mask + (int64_t)(y - 1) * w : NULL;
        int32_t *provRow = prov + (int64_t)y * w;
        int32_t *provPrevRow = (y > 0) ? prov + (int64_t)(y - 1) * w : NULL;
        for (int x = 0; x < w; x++) {
            if (!row[x]) {
                continue;
            }
            int32_t neigh[4];
            int nn = 0;
            if (x > 0 && row[x - 1]) {
                neigh[nn++] = provRow[x - 1];
            }
            if (prevRow != NULL && prevRow[x]) {
                neigh[nn++] = provPrevRow[x];
            }
            if (eightConn && prevRow != NULL && x > 0 && prevRow[x - 1]) {
                neigh[nn++] = provPrevRow[x - 1];
            }
            if (eightConn && prevRow != NULL && x + 1 < w && prevRow[x + 1]) {
                neigh[nn++] = provPrevRow[x + 1];
            }
            if (nn == 0) {
                provRow[x] = nextProv++;
            } else {
                int32_t lab = neigh[0];
                for (int k = 1; k < nn; k++) {
                    if (neigh[k] < lab) {
                        lab = neigh[k];
                    }
                }
                provRow[x] = lab;
                for (int k = 0; k < nn; k++) {
                    uf_union(parent, lab, neigh[k]);
                }
            }
        }
    }

    /* pass 2: resolve roots -> compact 1..N labels */
    int32_t *rootToFinal = (int32_t *)calloc((size_t)nextProv, sizeof(int32_t));
    if (rootToFinal == NULL) {
        free(prov);
        free(parent);
        return -1;
    }
    int32_t nextFinal = 1;
    for (int64_t i = 0; i < n; i++) {
        if (prov[i] == 0) {
            out_labels[i] = 0;
            continue;
        }
        int32_t root = uf_find(parent, prov[i]);
        int32_t fin = rootToFinal[root];
        if (fin == 0) {
            fin = nextFinal++;
            rootToFinal[root] = fin;
        }
        out_labels[i] = fin;
    }

    int32_t count = nextFinal - 1;

    free(prov);
    free(parent);
    free(rootToFinal);
    return count;
}
