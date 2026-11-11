#u23539195 coefficient justification: every coefficient in one list
import importlib

GROUPS = ["coeffs_segment","coeffs_lines","coeffs_recognise","coeffs_style",
          "coeffs_synth","coeffs_gantry","coeffs_dense"]


#---All: every registered coefficient, in group order---
def All():
    out = []
    for g in GROUPS:
        try:
            mod = importlib.import_module(g)
        except ImportError:
            continue
        out += list(mod.ENTRIES)
    #what each sweep actually showed is written after the runs and kept in its
    #own file, so the justification written beforehand stays as it was
    try:
        import observed
        for e in out:
            if e["key"] in observed.OBSERVED and not e.get("observed"):
                e["observed"] = observed.OBSERVED[e["key"]]
    except ImportError:
        pass
    return out


#---ByKey: one coefficient by its key---
def ByKey(key):
    for e in All():
        if e["key"] == key:
            return e
    raise KeyError(key)


#---Keys: every key---
def Keys():
    return [e["key"] for e in All()]
