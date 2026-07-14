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

## 2026-06-29 (Laptop)

- Loaded the repository from `C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot`.
- Read the previous Codex chat history from the pasted handoff attachment.
- Confirmed the laptop Codex session can access the repository files, including `PROJECT_CONTEXT.md`, `WORKLOG.md`, `README.md`, `Research/ReferenceLinks.txt`, the proposal PDF, and the scaffolded software/hardware/data folders.
- Read `PROJECT_CONTEXT.md`, `WORKLOG.md`, `README.md`, and `Research/ReferenceLinks.txt` to align with the PC session.
- Confirmed the persistent workflow: use GitHub/GitHub Desktop as the shared project state, and use `PROJECT_CONTEXT.md` plus `WORKLOG.md` as the cross-machine Codex handoff context.
- Noted that `git` is not currently available on this laptop shell's PATH, so GitHub Desktop should be used for commit/status/push actions unless Git is added to PATH.

## 2026-07-01 (PC)

- Created `Research/MODEL_REFERENCE_INVENTORY.md`.
- The inventory lists model families from `Research/ReferenceLinks.txt` as proposed models versus mentioned/baseline models.
- Added availability notes for code, training scripts, pretrained checkpoints, and datasets where verified.
- Identified TrOCR, One-DM, Sketch-RNN, Graves-style handwriting synthesis, and standard EMNIST/MNIST CNN/LeNet implementations as the most reusable externally available model/code references.
- Marked refs 8, 10, 12, 14, 16, and 17 for deeper PDF extraction because their model/baseline lists were not fully verified in this pass.
- Noted that external pretrained models should be treated as research references or benchmarks unless their use can be justified against the proposal constraint that core processing/design must be student-implemented and embedded.

## 2026-07-14 (PC)

- Decided on a gantry-based mechanical design for the handwriting robot.
- Designed several CAD parts for the gantry and started getting them 3D printed for physical testing.
- Obtained approximately half of the required store-bought gantry components.
- Current mechanical direction should be reflected in the first semester report progress section and used to guide the literature study discussion on XY motion, linear actuators, stepper motor control, pen lift/pressure, and positioning accuracy.

## Suggested Next Steps

- Commit and push `PROJECT_CONTEXT.md` and this `WORKLOG.md` using GitHub Desktop.
- Add a requirements traceability document that maps each proposal requirement to software, hardware, tests, and demonstration evidence.
- Begin planning the software architecture around the proposal modes: training, classification, and reproduction.
- Keep this worklog updated after each meaningful session before switching between PC and laptop.

# Commands run on the Odroid through SSH

- checked the IP of the odroid through Advanced IP Scanner app
- then ran ssh root@192.168.0.199 -> ip changes sometimes
- then log into odroid with linux and password 1234
-> GETTING PYTHON
- sudo apt update
- sudo apt install python3 python3-pip gpiod -y
