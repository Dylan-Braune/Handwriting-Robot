# Captions and provenance

The caption belongs below the figure in the report, set in bold, as
Appendix 4 requires; it is kept here rather than drawn onto the
picture.  The source line records which code and which data made the
figure.


## T15_software_block_diagram

T15 The three things the software does, and when. Preparation runs off the device and produces the weights and the writer profiles; reading a page and writing a new text are separate on-device requests that share those stored objects. The G-code file is the boundary: the synthesiser ends there and the gantry interpreter begins. The candidate search feeds the reader's and the writer model's scores back into generation, which is model scoring and carries no information about the physical page.

Source: `CleanPipeline3 module structure`

## T61_job_states

T61 The job states the server distinguishes and the paths between them. Generation and drawing are separate requests, so a trajectory can be inspected before the gantry moves, and a refusal is recorded distinctly from a failure. During motion the pulse loop checks an abort event and confirmed end stops on every iteration, and a trip latches a fault before the recovery routines run. No saved preview and no count of emitted pulses proves that the nib followed the intended path.

Source: `web/server.py job states; reproduce.RequestAbort / EmergencyStop`

## T59_page_vote

T59 Combining line evidence into one page decision. Each retained crop votes, weighted by how much ink it carries, so a sparse line cannot outweigh a full one; 20 of 20 individual crops agreed with the pooled result. Weighting by ink is also why crop rejection has to happen first: a dark artefact would otherwise carry a large weight. Illustration of Equation 3.30. The pooled probability and the agreement count are reported separately, and neither establishes accuracy without a reference.

Source: `recognise.ClassifyImage on stored crops | Author O page 1, author_shape_11_pw170_weights.npz`

## T60_round_trip

T60 The two comparisons that localise a fault. Re-rendering the emitted motion file and comparing it with the selected trajectory checks the conversion, and would reveal an inverted axis, a scale change or a missing stroke before anything is drawn. Comparing the physical page with that render checks execution instead. Panel (c) is a photograph of real robot output and is a different sentence from (a) and (b); neither comparison measures positional accuracy, which is qualified in Results.

Source: `synthesise + reproduce.WriteGcode / ParseGcode; photographs from ReportResults`
