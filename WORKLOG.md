# Worklog

Use this file as a short handoff record between PC, laptop, and Codex sessions. Keep entries dated, note the machine in brackets, and list only meaningful project changes or decisions.

## 2026-06-29 (PC)

- Confirmed Codex can access the local repository at `C:\Users\user-pc\Documents\GitHub\Handwriting Robot`.
- Confirmed the repository tracks GitHub remote `https://github.com/Dylan-Braune/Handwriting-Robot.git` on branch `main`.
- Reviewed the current repository structure:
  - `Data/Datasets`
  - `Data/Models`
  - `Hardware/CAD`
  - `Hardware/Simulations`
  - `Research`
  - `Software/CNN`
  - `Software/ComputerVision`
  - `Software/GantryControl`
  - `Software/WebDisplay`
- Read `README.md`, which states that the proposal PDF must always be followed and sources must be provided for researched/explained content.
- Read `Research/ReferenceLinks.txt` and `Data/Datasets/DatasetIdeas.txt`.
- Extracted the filled proposal content from `ProjectProposal2026DylanFinalRev0 - Dylan Braune-1.pdf` form fields.
- Identified the proposal PDF as the source of truth for project requirements, specifications, limitations, demonstrations, field conditions, and student design contributions.
- Added `PROJECT_CONTEXT.md` as a checked-in summary of the proposal requirements and working rules for future Codex sessions.
- Confirmed the major proposal constraints:
  - Processing must run on an embedded/single-board computer, not a PC platform.
  - Library functions may be used for hardware interfacing only.
  - Minimum reproduction target is 85% character accuracy and 85% stylistic correlation.
  - Classification target is 95% overall accuracy for writer and text recognition.
  - Reproduction speed must be no slower than 4 seconds per character.
  - Classification timing target is 25 ms per character plus 2 seconds for image capture and conditioning.
  - Gantry absolute positional error must not exceed 0.5 mm.
  - System must support at least 10 trained user styles with permanent storage of trajectory mappings.
  - Field conditions are 400-1000 lux lighting and a stable, level, vibration-free surface.
- Committed `PROJECT_CONTEXT.md` locally with commit `940597f Add project context summary`.
- Push from Codex hung due to likely credential/network handoff, so GitHub Desktop is preferred for committing and pushing from this point.

## Suggested Next Steps

- Commit and push `PROJECT_CONTEXT.md` and this `WORKLOG.md` using GitHub Desktop.
- Add a requirements traceability document that maps each proposal requirement to software, hardware, tests, and demonstration evidence.
- Begin planning the software architecture around the proposal modes: training, classification, and reproduction.
- Keep this worklog updated after each meaningful session before switching between PC and laptop.
