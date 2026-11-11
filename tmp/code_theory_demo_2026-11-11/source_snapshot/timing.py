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


#----------------------------------------------------------------------
#ALLOWANCE: the time the proposal allows, as a formula not a constant
#both requirements scale with how much writing is involved, so a bare
#total cannot be marked against them: a page holding twice the text is
#allowed twice as long.  the character count comes from the work itself,
#the text read back for a classification and the text asked for when
#reproducing
#----------------------------------------------------------------------
ALLOWANCE = {
    #reading a photographed page: a fixed cost for the page, then per character
    "classify": lambda chars: 2.0+0.025*chars,
    #generating a line and drawing it: per character, the pen being the slow part
    "reproduce": lambda chars: 4.0*chars,
    #a drawing-only run is judged on the same reproduction allowance
    "draw": lambda chars: 4.0*chars,
}

#---AllowanceText: the formula itself, so a reader can check the sum---
ALLOWANCE_TEXT = {
    "classify": "2 s + 0.025 s per character",
    "reproduce": "4 s per character",
    "draw": "4 s per character",
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
        #seconds recorded by Add rather than measured here: work that happened
        #in an earlier request and still counts towards this one.  drawing a
        #line is asked for separately from generating it, but the requirement
        #covers the whole reproduction, so generation is carried across
        self.external = 0.0
        self.t0 = time.perf_counter()

    #---Stage: time one step, naming the characters it covered---
    def Step(self, name, chars=None):
        return Stage(self, name, self.chars if chars is None else chars)

    #---Record: store one finished step---
    def Record(self, name, wall, chars):
        self.stages.append(dict(stage=name, seconds=round(wall, 4),
                                characters=chars,
                                secondsPerChar=round(wall/chars, 5) if chars else None))

    #---Add: a stage timed inside this run but measured by the code itself---
    #the wall clock already covers it, so it must not be added again
    def Add(self, name, seconds, chars=0):
        self.Record(name, float(seconds), chars)

    #---Carry: a stage measured in an EARLIER request that counts here too---
    #this one is outside this run's wall clock, so it is added to the total
    def Carry(self, name, seconds, chars=0):
        self.external += float(seconds)
        self.Record(name, float(seconds), chars)

    #---Total: everything, including whatever was not inside a stage---
    def Total(self):
        return (time.perf_counter()-self.t0)+self.external

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
        formula = ALLOWANCE.get(self.kind)
        if formula and self.chars:
            limit = formula(self.chars)
            out["limitSeconds"] = round(limit, 2)
            out["limitFormula"] = ALLOWANCE_TEXT.get(self.kind, "")
            out["withinLimit"] = bool(total <= limit)
            out["fractionOfLimit"] = round(total/limit, 3)
            out["secondsSpare"] = round(limit-total, 2)
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
            lines.append("  allowed %.2f s (%s for %d characters): %s, %.2f s %s"
                         % (d["limitSeconds"], d["limitFormula"], d["characters"],
                            "MET" if d["withinLimit"] else "EXCEEDED",
                            abs(d["secondsSpare"]),
                            "to spare" if d["secondsSpare"] >= 0 else "over"))
        if d["secondsPerCharacter"]:
            lines.append("  %.4f s per character, %.2f characters per second"
                         % (d["secondsPerCharacter"], d["charactersPerSecond"]))
        return "\n".join(lines)
