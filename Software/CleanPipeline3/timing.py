"""Stage timings, and the per-character figures the requirements are stated in.

Every request that does real work builds one Run, opens a Stage around each
step, and hands the whole thing back to the browser as json.  Two things are
measured and they are not the same:

  wall     how long the step took, start to finish
  perChar  that time divided by the number of characters involved

The requirements are written per page and per character, so a bare total
cannot be checked against them.  A page that takes twice as long because it
holds twice as much writing has not got slower, and only the per-character
figure shows that.

Nothing here imports the pipeline, so it can be used from the server, from a
script, or from a test without dragging the models in.
"""
import json
import time


#---LIMITS: what the proposal asks for, so a measurement can be marked---
#R3 and R4 are the timing requirements; the numbers are seconds
LIMITS = {
    "classify": 30.0,      #R4: a photographed page read and named
    "reproduce": 120.0,    #R3: a line generated and drawn
}


class Stage:
    """one timed step, used as a with block"""

    def __init__(self, run, name, chars=0):
        self.run = run
        self.name = name
        self.chars = chars
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter()-self.t0
        self.run.Record(self.name, wall, self.chars)
        return False


class Run:
    """the timings for one whole request"""

    def __init__(self, kind, chars=0, lines=0):
        self.kind = kind
        self.chars = chars
        self.lines = lines
        self.stages = []
        self.t0 = time.perf_counter()

    #---Stage: time one step, naming the characters it covered---
    def Step(self, name, chars=None):
        return Stage(self, name, self.chars if chars is None else chars)

    #---Record: store one finished step---
    def Record(self, name, wall, chars):
        self.stages.append(dict(stage=name, seconds=round(wall, 4),
                                characters=chars,
                                secondsPerChar=round(wall/chars, 5) if chars else None))

    #---Note: a measurement that was taken rather than timed---
    def Add(self, name, seconds, chars=0):
        self.Record(name, float(seconds), chars)

    #---Total: everything, including whatever was not inside a stage---
    def Total(self):
        return time.perf_counter()-self.t0

    #---AsDict: the shape the browser reads---
    def AsDict(self):
        total = self.Total()
        accounted = sum(s["seconds"] for s in self.stages)
        out = dict(
            kind=self.kind,
            characters=self.chars,
            lines=self.lines,
            totalSeconds=round(total, 3),
            measuredSeconds=round(accounted, 3),
            #whatever the stages did not account for: encoding, json, disk
            overheadSeconds=round(max(0.0, total-accounted), 3),
            secondsPerCharacter=round(total/self.chars, 4) if self.chars else None,
            secondsPerLine=round(total/self.lines, 3) if self.lines else None,
            charactersPerSecond=round(self.chars/total, 2) if self.chars and total else None,
            stages=self.stages,
        )
        limit = LIMITS.get(self.kind)
        if limit:
            out["limitSeconds"] = limit
            out["withinLimit"] = bool(total <= limit)
            out["fractionOfLimit"] = round(total/limit, 3)
        return out

    def AsJson(self):
        return json.dumps(self.AsDict(), indent=1)

    #---Report: the same thing as lines of text, for a terminal---
    def Report(self):
        d = self.AsDict()
        width = max([len(s["stage"]) for s in d["stages"]] or [8])
        lines = ["timing for %s: %.2f s total, %d characters, %d lines"
                 % (d["kind"], d["totalSeconds"], d["characters"], d["lines"])]
        for s in d["stages"]:
            share = 100*s["seconds"]/d["totalSeconds"] if d["totalSeconds"] else 0
            perChar = "" if not s["secondsPerChar"] else "  %.4f s/char" % s["secondsPerChar"]
            lines.append("  %-*s %8.3f s  %5.1f%%%s"
                         % (width, s["stage"], s["seconds"], share, perChar))
        if d["overheadSeconds"] > 0.001:
            lines.append("  %-*s %8.3f s" % (width, "unaccounted", d["overheadSeconds"]))
        if d.get("limitSeconds"):
            lines.append("  requirement %.0f s: %s (%.0f%% of the allowance)"
                         % (d["limitSeconds"], "met" if d["withinLimit"] else "EXCEEDED",
                            100*d["fractionOfLimit"]))
        if d["secondsPerCharacter"]:
            lines.append("  %.4f s per character, %.2f characters per second"
                         % (d["secondsPerCharacter"], d["charactersPerSecond"]))
        return "\n".join(lines)
