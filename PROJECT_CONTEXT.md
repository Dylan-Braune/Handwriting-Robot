# Project Context

This repository is for Dylan Braune's handwriting robot project. The source of truth for project scope, requirements, specifications, limitations, and demonstrations is:

- `ProjectProposal2026DylanFinalRev0 - Dylan Braune-1.pdf`

When project decisions conflict with this summary, follow the proposal PDF and update this summary afterward.

## Core Project Goal

Build a handwriting robot that can train on multiple users' handwritten text, classify handwritten text by content and writer, and reproduce arbitrary digital text on paper in a selected trained writer's handwriting style.

The system has three main modes:

- Training: scan handwritten text, condition the image, separate characters or connected character groups, train the neural network, create trajectory maps, and store user-specific mappings.
- Classification: scan handwritten text, condition and segment it, then identify the text content and writer.
- Reproduction: retrieve stored trajectory maps for the desired writer and text, convert them to motor commands, and write using a 2D gantry and pen lift mechanism.

## Non-Negotiable Proposal Constraints

- Processing must be done on an embedded/single-board computer, not on a PC platform.
- Library functions may be used for hardware interfacing only.
- Image processing, character separation, neural network design/training, pen trajectory mapping, gantry design, and motion control are intended student design contributions.
- Off-the-shelf components are allowed for physical motors, camera, camera driver, single-board computer, stepper motor driver, pen, storage, display, and mechanical gantry components.

## Mission Requirements

1. Reproduce any desired digital text on paper in the writing style of any writer from the training set.
   - Target: at least 85% character accuracy and 85% stylistic correlation to the chosen writer.
   - Demonstration: reproduce a sentence in a specific user style; at least 85% of characters must visibly match the input text and chosen style.

2. Decode handwritten text content and identify which writer from the training set wrote it.
   - Target: 95% overall accuracy for written text and writer identification.
   - Demonstration: classify a predefined handwritten paragraph from a training-set writer; classified text and writer must reach at least 95% overall accuracy.

3. Reproduce text at a practical near-real-time speed.
   - Target: no slower than 4 seconds per character.
   - Demonstration: reproduce a sentence of `x` characters within `4*x` seconds.

4. Classify handwritten text in a practical amount of time.
   - Target: no slower than 25 ms per character, plus 2 seconds for image capture and conditioning.
   - Demonstration: classify a sentence of `x` handwritten characters within `2 + x*0.025` seconds.

5. Draw desired character trajectory maps with precise gantry motion.
   - Target: absolute positional error must not exceed 0.5 mm.
   - Demonstration: recreate the same word twice without changing dataset characters; overlaid words should appear identical and stay within 0.5 mm absolute positional error.

6. Train on and store multiple writing styles.
   - Target: at least 10 different users' writing styles, with permanent storage of necessary character mappings for all 10 users.
   - Demonstration: display the number of users in the system, select a random user out of 10, and reproduce text in that user's style.

## Real-World Field Conditions

- Operate in well-lit lab or office conditions between 400 and 1000 lux to reduce image-capture noise.
- Operate on a stable, level, vibration-free surface so page scanning and pen motion remain accurate.

## Current Repository Shape

- `Data/Datasets`: dataset notes and future dataset handling.
- `Data/Models`: trained model artifacts or model metadata.
- `Hardware/CAD`: gantry and fixture CAD work.
- `Hardware/Simulations`: mechanical, control, or motion simulations.
- `Research`: proposal references, papers, and source notes.
- `Software/CNN`: neural network implementation and experiments.
- `Software/ComputerVision`: image conditioning, segmentation, and trajectory extraction.
- `Software/GantryControl`: stepper control, path execution, and pen lift control.
- `Software/WebDisplay`: user interface or display layer.

## Research Starting Points

The repository includes `Research/ReferenceLinks.txt` and one local research PDF. The proposal references these sources directly:

- Babushkin et al., "Analyzing handwriting legibility through hand kinematics," Frontiers in Artificial Intelligence, 2025.
- Agduk and Aydemir, "Classification of Handwritten Text Signatures by Person and Gender," Acta Informatica Pragensia, 2022.
- O. Teikari, "Reproducing Handwriting with Position-Controlled Robots," Master's thesis, Tampere University of Technology, 2016.
- T. Huang and R. Xiong, "Cost-Effective Robotic Handwriting System with AI Integration," IEEE LISAT, 2024.

Potential datasets already noted in the repository:

- EMNIST: 28x28 pixel handwritten digit and character images.
- Kaggle digit/character dataset: 28x28 images for digits and A-Z-style character classes.

## Working Rules For Future Codex Sessions

- Read this file, the proposal PDF, `README.md`, and `Research/ReferenceLinks.txt` before making design decisions.
- Treat measurable proposal requirements as acceptance criteria for software, hardware, and tests.
- If a requirement appears impossible, document the reason, evidence, trade-offs, and proposed justification before changing scope.
- Keep sources attached to researched claims, as requested in `README.md`.
- Commit meaningful changes to Git and push to GitHub so the PC and laptop can stay synchronized.
