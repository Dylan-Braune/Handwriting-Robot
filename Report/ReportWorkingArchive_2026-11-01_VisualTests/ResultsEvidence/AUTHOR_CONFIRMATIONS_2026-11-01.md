# Author confirmations, 1 November 2026

Selected demonstration: library-trained joint reader and library-trained eleven-writer pw170 classifier, width 0.17. NumPy-trained models are compared separately.

The supplied numerical-writer console log names author_shape_11w_pw170_scratch_weights.npz, 1031 train/155 validation lines, initial learning rate about 0.001 with cosine decay, best displayed validation accuracy 0.968 by epoch 41, about 19--23 minutes/epoch, and continues to epoch 47 before an incomplete traceback. A 50-epoch budget was requested; completion is not inferred. The author reports float32 training. No precision ablation proves why the trained-model scores differ.

The numerical reader excerpt records 6481 Teklia + 569 personal training lines and 976 Teklia + 100 personal validation lines, resumed at epoch 7 from an existing numerical checkpoint. Epoch 16 displayed public validation character accuracy 92.89%, personal 93.09%, and best validation loss 11.663. Reported epoch durations are 111.3--122.3 minutes. This excerpt does not establish the final epoch or map its last epoch to the evaluated checkpoint.

The library sweep is on CUDA, 1031 train/155 validation lines and eleven classes. Widths 0.06/0.09/0.11/0.17 each have 50-epoch budgets, best accuracies 0.9806/0.9742/0.9806/0.9935 at epochs 38/43/39/43. These are the Design's existing sweep values, now supported by a supplied console record.

The author confirms the seven reported side readings were taken on the final calibrated gantry, using a metal non-digital vernier caliper; identification/least count are pending. More readings and repeated-word overlays are planned but are not invented. The author observes operation over a wider lighting range, but has not supplied numeric lux readings.

The author confirms signed consent from all five personal contributors. Institutional ethics status is still requested separately. Consent signatures are retained privately.

Approximate current budget R4000. ODROID borrowed at no cost, to be returned after the project. Subsequently listed clipboard R90, paper R130, gel pen R30 and estimated training electricity R500 total R750. Inclusion in approximate total is unconfirmed. Ledger photograph is preserved separately; unclear handwritten totals are not treated as exact accounting.

The author considers Claude's retained timing runs complete and permits removal of stopped jobs. The review retains the narrower explicit-completion subset for primary reporting, because one elapsed-time-retained run has an explicit STOPPED marker. Maximum timing is unchanged.
