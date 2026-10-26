"""Flatten an AcroForm PDF by drawing each widget's normal appearance stream
into the page content, so that pdfpages' \\includepdf renders it.

\\includepdf imports page content but not annotations, which is why an
unflattened form prints as empty boxes.  Each widget's value is already drawn
in its /AP /N appearance stream; this moves that stream into the page itself,
positioned per PDF 32000-1 section 12.5.5.

Usage:  python -I flatten_form.py <in.pdf> <out.pdf>
"""
import sys

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           NameObject, NumberObject)


def appearance_matrix(bbox, matrix, rect):
    """The cm matrix that maps the appearance BBox onto the widget Rect."""
    bx0, by0, bx1, by1 = [float(v) for v in bbox]
    a, b, c, d, e, f = [float(v) for v in matrix]
    corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
    xs, ys = [], []
    for x, y in corners:
        xs.append(a * x + c * y + e)
        ys.append(b * x + d * y + f)
    tx0, tx1 = min(xs), max(xs)
    ty0, ty1 = min(ys), max(ys)

    rx0, ry0, rx1, ry1 = [float(v) for v in rect]
    rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
    ry0, ry1 = min(ry0, ry1), max(ry0, ry1)

    tw, th = tx1 - tx0, ty1 - ty0
    sx = (rx1 - rx0) / tw if tw else 1.0
    sy = (ry1 - ry0) / th if th else 1.0
    return sx, 0.0, 0.0, sy, rx0 - tx0 * sx, ry0 - ty0 * sy


def normal_appearance(annot):
    """The /AP /N form XObject, resolving the /AS sub-dictionary if present."""
    ap = annot.get("/AP")
    if ap is None:
        return None
    n = ap.get("/N")
    if n is None:
        return None
    n = n.get_object()
    if isinstance(n, DictionaryObject) and "/BBox" not in n:
        # a sub-dictionary of states (checkboxes, radio buttons)
        state = annot.get("/AS")
        if state is None:
            return None
        n = n.get(state)
        n = n.get_object() if n is not None else None
    if n is None or "/BBox" not in n:
        return None
    return n


def flatten(src, dst):
    reader = PdfReader(src)
    writer = PdfWriter(clone_from=reader)

    drawn = skipped = 0
    for page in writer.pages:
        annots = page.get("/Annots")
        if not annots:
            continue

        resources = page.get("/Resources")
        if resources is None:
            resources = DictionaryObject()
            page[NameObject("/Resources")] = resources
        xobjects = resources.get("/XObject")
        if xobjects is None:
            xobjects = DictionaryObject()
            resources[NameObject("/XObject")] = xobjects

        ops = []
        for i, ref in enumerate(annots):
            annot = ref.get_object()
            ap = normal_appearance(annot)
            rect = annot.get("/Rect")
            if ap is None or rect is None:
                skipped += 1
                continue
            # An appearance stream is a form XObject, but many writers leave
            # /Type and /Subtype implicit.  Once it is referenced from page
            # /Resources both pypdf and pdfTeX require them explicitly.
            if "/Subtype" not in ap:
                ap[NameObject("/Subtype")] = NameObject("/Form")
            if "/Type" not in ap:
                ap[NameObject("/Type")] = NameObject("/XObject")
            name = NameObject(f"/FlatFld{i}")
            xobjects[name] = ap.indirect_reference or ap
            m = appearance_matrix(ap.get("/BBox"),
                                  ap.get("/Matrix", [1, 0, 0, 1, 0, 0]),
                                  rect)
            ops.append("q {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} cm {} Do Q"
                       .format(*m, name))
            drawn += 1

        if not ops:
            continue

        extra = DecodedStreamObject()
        # a leading "q ... Q" guard keeps the page's own graphics state intact
        extra.set_data(("\nq\n" + "\n".join(ops) + "\nQ\n").encode("latin-1"))
        extra_ref = writer._add_object(extra)

        contents = page.get("/Contents")
        if contents is None:
            page[NameObject("/Contents")] = extra_ref
        elif isinstance(contents.get_object(), ArrayObject):
            contents.get_object().append(extra_ref)
        else:
            page[NameObject("/Contents")] = ArrayObject(
                [contents.indirect_reference or contents, extra_ref])

        # the appearances are now page content, so drop the widgets
        del page[NameObject("/Annots")]

    # remove the now-redundant form dictionary
    root = writer._root_object
    if "/AcroForm" in root:
        del root[NameObject("/AcroForm")]

    with open(dst, "wb") as fh:
        writer.write(fh)
    print(f"drew {drawn} appearance streams, skipped {skipped}")


if __name__ == "__main__":
    flatten(sys.argv[1], sys.argv[2])
