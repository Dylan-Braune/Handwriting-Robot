from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignResultsDiscussion_2026-11-01'
p=B/'3_Main_Report/4_Results.tex';s=p.read_text()
changes={
'The evaluated reader reaches 94.39\\% character accuracy on 1\\,076 validation lines. The writer point estimate exceeds 95\\%; the pooled text result is 0.61 percentage points below it.':'The selected joint reader reaches 94.06\\% character accuracy on 1\\,076 validation lines; the numerical comparison reader reaches 94.39\\%. The writer point estimate exceeds 95\\%; the selected reader is 0.94 percentage points below it.',
'the latter supplies the headline 94.39\\% result.':'the selected joint reader supplies the 94.06\\% system result and the scratch reader supplies a 94.39\\% comparison result.',
'The reader validation gives $100(1-2567/45747)=94.3887\\%$.':'The selected reader gives $100(1-2719/45747)=94.0564\\%$, while the numerical reader gives $100(1-2567/45747)=94.3887\\%$.',
'The measured difference supports evaluating the scratch reader as a final reader candidate without changing the provenance of the earlier joint-reader Design traces.':'The author selects the library-trained joint reader for the demonstration. The comparison therefore supports the numerical implementation without substituting its result for the selected system.',
'the strongest photographed writer\'s character score is 97.70\\%.':'the strongest photographed writer\'s character score under the selected reader is 95.81\\%.',
'The pooled text score is 0.61 percentage points below 95\\%;':'The selected reader\'s pooled text score is 0.94 percentage points below 95\\%;',
'Public and photographed character scores differ by only 0.03 percentage points, although their word-match scores differ by 2.93 points. Exactly 242/1\\,076 reader lines are recovered with no character edits.':'The selected reader gives 94.12\\% on the public subset and 93.33\\% on photographed lines, and recovers 221/1\\,076 complete lines without an edit. The numerical reader\'s corresponding character scores are 94.39\\% and 94.42\\%, with 242 exact lines. The selected and numerical readers have pooled positional word-match scores of 72.16\\% and 73.47\\%, respectively.',
'a 150~mm caliper, with a reported 0.01~mm display resolution;':'a metal non-digital vernier caliper, whose make, model and least count remain to be recorded;',
'Caliper display resolution is kept distinct from measurement accuracy on finite-width pen marks.':'Caliper least count is not assumed from the number of digits typed in a result; reading uncertainty also depends on finite-width pen marks.',
'width=0.83\\linewidth]{Figures/results_physical_examples.pdf}':'width=0.98\\linewidth]{Figures/results_physical_examples.pdf}',
}
for a,b in changes.items():
 if a not in s:print('not found',a[:80])
 s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
p=B/'3_Main_Report/design/3_5_classification.tex';s=p.read_text()
s=s.replace("The reader checkpoint is \\texttt{paper\\_cnn\\_bilstm\\_ctc\\_joint\\_best.npz}; the selected eleven-writer checkpoint is \\texttt{author\\_shape\\_11\\_pw170\\_weights.npz}.","The selected demonstration reader is \\texttt{paper\\_cnn\\_bilstm\\_ctc\\_joint\\_best.npz}; the selected eleven-writer checkpoint is \\texttt{author\\_shape\\_11\\_pw170\\_weights.npz}. Results in Section~\\ref{sec:results} compare these selected models with separately trained numerical-route reader and writer checkpoints. The joint reader also supplies the numerical Design traces, so those traces retain their original checkpoint provenance.")
s=s.replace('Confirm applicability of the archived comparison to the submitted implementation and add the matched final training comparison. Obtain any missing final reader/writer comparisons from the existing project work.','Confirm applicability of the archived fixed-parameter comparison to the submitted implementation. The new common-material trained-model comparison is given in Table~\\ref{tab:dis-training}; it is distinct from a controlled equal-initialisation training experiment.')
p.write_text(s,encoding='utf-8')
# Preserve a concise, faithful record of the newly supplied author information.
(B/'ResultsEvidence/AUTHOR_CONFIRMATIONS_2026-11-01.md').write_text('''# Author confirmations, 1 November 2026

Selected demonstration: library-trained joint reader and library-trained eleven-writer pw170 classifier, width 0.17. NumPy-trained models are compared separately.

The supplied numerical-writer console log names author_shape_11w_pw170_scratch_weights.npz, 1031 train/155 validation lines, initial learning rate about 0.001 with cosine decay, best displayed validation accuracy 0.968 by epoch 41, about 19--23 minutes/epoch, and continues to epoch 47 before an incomplete traceback. A 50-epoch budget was requested; completion is not inferred. The author reports float32 training. No precision ablation proves why the trained-model scores differ.

The numerical reader excerpt records 6481 Teklia + 569 personal training lines and 976 Teklia + 100 personal validation lines, resumed at epoch 7 from an existing numerical checkpoint. Epoch 16 displayed public validation character accuracy 92.89%, personal 93.09%, and best validation loss 11.663. Reported epoch durations are 111.3--122.3 minutes. This excerpt does not establish the final epoch or map its last epoch to the evaluated checkpoint.

The library sweep is on CUDA, 1031 train/155 validation lines and eleven classes. Widths 0.06/0.09/0.11/0.17 each have 50-epoch budgets, best accuracies 0.9806/0.9742/0.9806/0.9935 at epochs 38/43/39/43. These are the Design's existing sweep values, now supported by a supplied console record.

The author confirms the seven reported side readings were taken on the final calibrated gantry, using a metal non-digital vernier caliper; identification/least count are pending. More readings and repeated-word overlays are planned but are not invented. The author observes operation over a wider lighting range, but has not supplied numeric lux readings.

The author confirms signed consent from all five personal contributors. Institutional ethics status is still requested separately. Consent signatures are retained privately.

Approximate current budget R4000. ODROID borrowed at no cost, to be returned after the project. Subsequently listed clipboard R90, paper R130, gel pen R30 and estimated training electricity R500 total R750. Inclusion in approximate total is unconfirmed. Ledger photograph is preserved separately; unclear handwritten totals are not treated as exact accounting.

The author considers Claude's retained timing runs complete and permits removal of stopped jobs. The review retains the narrower explicit-completion subset for primary reporting, because one elapsed-time-retained run has an explicit STOPPED marker. Maximum timing is unchanged.
''',encoding='utf-8')
print('Selected-model and author-confirmation updates applied.')
