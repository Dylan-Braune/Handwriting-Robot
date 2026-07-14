# Model Reference Inventory

Date checked: 2026-07-01  
Machine: PC

This file inventories the model families found in `Research/ReferenceLinks.txt`, separating models proposed by the referenced paper from models only mentioned as baselines, comparisons, or related work. It also records whether code, training scripts, pretrained checkpoints, or datasets appear to be available.

Availability labels:

- **Official code**: code is linked by the paper/authors or hosted by the named project.
- **Community code**: usable third-party implementations exist, but not necessarily from the paper authors.
- **Checkpoints available**: trained weights are publicly linked.
- **No code found**: no official implementation/checkpoint found during this pass.
- **Dataset available**: dataset is publicly accessible or documented.

## High-Level Takeaways

- Most recognition references propose CNN, CNN-SVM, CNN-RNN, CNN-LSTM, or CNN-RNN-CTC-style models.
- Most paper-specific recognition models do **not** appear to publish official training code or trained weights.
- The strongest immediately reusable code/checkpoint candidates are:
  - TrOCR: official Microsoft/unilm code and Hugging Face checkpoints for IAM handwritten OCR and SROIE printed OCR.
  - One-DM: official PyTorch code, training scripts, English datasets, and pretrained checkpoints for handwriting generation.
  - Sketch-RNN: official Magenta code, training instructions, QuickDraw-compatible datasets, and many pretrained sketch models.
  - Graves-style RNN handwriting synthesis: community TensorFlow implementation with included pretrained model/checkpoints.
  - Standard MNIST/EMNIST CNN/LeNet models: abundant community code, datasets available, but paper-specific weights usually absent.

## Reference-by-Reference Inventory

| Ref | Paper / source topic | Proposed model(s) in that reference | Mentioned / baseline / comparison models | Dataset(s) used or mentioned | Code / training / pretrained availability |
|---|---|---|---|---|---|
| 1 | Handwritten Text Classification Based on Convolutional Neural Network | Custom CNN with four convolutional layers and one dense layer; LeNet-5 transfer-learned model | HMM, SVM, incremental recognition, part-based methods, ensemble methods, neural networks, Decision Tree, Random Forest, MLP, CNN, Siamese CNN, deep 9-layer CNN | EMNIST Balanced, EMNIST By_Class; mentions MNIST, COREL1000, ORL/face datasets, LFW, food image dataset, ECG datasets | No official paper code/checkpoints found. EMNIST is available from NIST/TensorFlow/Kaggle. LeNet/CNN training code is widely available as community code. |
| 2 | Programming ABB Industrial Robot for an Accurate Handwriting | ABB industrial robot programming workflow using CAD-created letter paths and RobotStudio/RAPID-style programming | CAD modeling in SolidWorks; robot workstation/workobject setup; ResearchGate snippet also mentions polynomial backlash-error modeling in related text | Not a learning dataset paper | No ML model code/checkpoints. The relevant artifacts would be CAD files and robot programs; no official downloadable code found. |
| 3 | Cost-Effective Robotic Handwriting System with AI Integration | Low-cost handwriting machine using Raspberry Pi Pico, 3D-printed mechanics, and a TensorFlow.js handwriting generation model that converts text to stroke trajectories | TensorFlow.js; Graves recurrent handwriting generation; Sketch-RNN; One-DM diffusion handwriting generation; AccelStepper / Arduino-style control references | User-supplied text/stroke trajectories; no specific public training dataset for the paper's own TF.js model found | No official repository for this paper's own system found. Related models have code: Sketch-RNN official Magenta code and pretrained QuickDraw models; One-DM official code and checkpoints; Graves-style RNN community code/checkpoints. |
| 4 | Handwriting-Based Gender Classification Using Robotic and Machine Learning Models | Robotic-model-derived kinematic/dynamic feature pipeline; LPC and SSA features; machine learning gender classifiers | Prior gender/age handwriting systems including CNNs, transfer learning, random forests, kernel discriminant analysis, SVM, RF, KNN, XGBoost, gradient boosting/AdaBoost-style methods from related literature | BiosecurID, PaHaW | No official code/checkpoints found. Dataset access may require dataset owner/license approval. |
| 5 | Analyzing handwriting legibility through hand kinematics | Temporal Convolutional Network (TCN) / deep learning legibility classifier using stylus plus hand kinematics; SHAP/Shapley explainability | Stylus-only model vs stylus-plus-hand model; statistical feature analysis | Study-specific handwriting/kinematics data labeled by experts | No official code/checkpoints found. Paper is open access; dataset/checkpoint availability not found in this pass. |
| 6 | Reproducing Handwriting with Position-Controlled Robots | Position-controlled robotic handwriting reproduction / trajectory mapping approach | Robot motion/path planning, position control, handwriting trajectory generation | Thesis-specific handwriting/robot trajectory data | No official code/checkpoints found. Useful as methodology for trajectory mapping and speed/accuracy justification rather than a reusable ML model. |
| 7 | Intelligent Handwritten Recognition Using Hybrid CNN Architectures Based-SVM Classifier with Dropout | Hybrid CNN-SVM architecture with dropout; automatic feature extraction/classification; M3CE/cross-entropy training-rule discussion | SVM, CNN, dropout, max-margin minimum classification error, cross-entropy; state-of-the-art Arabic HTR systems | AHDB, AHCD, HACDB, IFN/ENIT | No official code/checkpoints found. Paper full text is available. Datasets vary in public/licensed availability; IFN/ENIT is commonly used for Arabic handwriting research but is not a plug-and-play open checkpoint. |
| 8 | Pixel / image-size reference for recognition | Not enough verified model detail from the linked PDF during this pass | Likely handwritten character/digit recognition models; needs direct PDF extraction in a later pass | Not fully verified | Marked for follow-up. |
| 9 | A Study on a Hybrid CNN-RNN Model for Handwritten Recognition Based on Deep Learning | Hybrid CNN-RNN model: ResNet-34-style CNN feature extractor plus Bidirectional LSTM sequence modeling | Single CNN baseline; ANN, CNN, LSTM/RNN, CNN-RNN, seq2seq CNN-RNN, generated-sample augmentation, Devanagari CNN, TDNN-SVM, LSTM-based online HWR, DNN depth-sensor air-writing, tree-BLSTM math recognition | IAM handwriting dataset | No official code/checkpoints found. Uses ResNet-34/ImageNet transfer-learning concepts, which are available in standard frameworks. IAM dataset requires access/license handling. |
| 10 | Handwritten Text Recognition Using Deep Learning: A CNN-LSTM Approach | CNN-LSTM handwritten text recognition approach | CNN, LSTM, likely CTC-style HTR components depending on full paper details | Not fully verified from article page | No official code/checkpoints found in quick pass. Needs PDF extraction follow-up. |
| 11 | Classification of Non-native Handwritten Characters Using CNN | Custom CNN for HIEC with five convolutional layers, max-pooling layers, one hidden layer, and dropout | AlexNet, VGGNet, GoogLeNet, ResNet50, InceptionV3, MobileNet, DenseNet121; other HCR models from Mor et al., Weng and Xia, Joshi and Risodkar | HIEC dataset: 16,496 images from 260 people, after augmentation; handwritten isolated English characters | No official code/checkpoints found. HIEC dataset public download was not found in this pass. Baseline architectures are available in PyTorch/TensorFlow with ImageNet weights, but those are not handwriting-specific checkpoints. |
| 12 | Training / characters / epochs reference | Not fully verified from linked IOP PDF during this pass | Likely CNN/HTR training comparisons | Not fully verified | Marked for follow-up. |
| 13 | CNN-RNN Based Handwritten Text Recognition | CNN-RNN model for offline handwritten text recognition; CNN feature extractor + RNN/LSTM sequence model + CTC decoding, based on available abstract/summary | CNN, RNN, LSTM, CTC | Paper mentions connected/cursive Sinhala-word style recognition in available snippets; exact dataset needs full PDF extraction | No official code/checkpoints found. |
| 14 | Types of neural networks reference | Survey/reference-style source, not a single proposed handwriting model | Likely neural-network families used for handwriting/recognition; needs direct PDF extraction | Not fully verified | Marked for follow-up. |
| 15 | A Novel Handwritten Digit Classification System Based on CNN Approach | Proposed CNN with three convolutional hidden blocks using ReLU, batch normalization, max-pooling, dropout, dense layers, RMSprop, data augmentation, ERF-based filter selection; reports 99.98% on MNIST and 99.40% on noisy MNIST | CNN, DCST, LIRA, SNN, ASSOM, CNN/SVM, OPF, DM, LeNet5/SVM, unsupervised deep convolutional SOM, single-hidden-layer feedforward network | MNIST; noisy MNIST with additive white Gaussian noise; augmented MNIST | No official GitHub/checkpoints found in paper page. MNIST is openly available and many community CNN training scripts exist. |
| 16 | CNN structure / features / speed reference | Not fully verified from ACM page during this pass | CNN architecture and speed-related feature extraction likely | Not fully verified | Marked for follow-up. |
| 17 | Recognition accuracies of other methods | Accuracy-comparison/reference source; exact model list needs direct PDF extraction | Likely multiple HWR/OCR models used for comparison | Not fully verified | Marked for follow-up. |

## Reusable Model Candidates For This Project

These are not all proposal-compliant as-is. The project proposal says processing must run on an embedded/single-board computer and library functions are only for hardware interfacing. These external models are therefore best used for research benchmarking, architecture inspiration, dataset comparison, or justification unless we implement the required project components ourselves.

| Model / family | Task fit | Code availability | Training code | Pretrained weights | Dataset links / notes | Suitability notes |
|---|---|---|---|---|---|---|
| Custom EMNIST CNN / LeNet-5 | Character classification | Community code abundant; no official Ref 1 repo found | Yes, easy to implement from scratch | Community checkpoints may exist, but not paper-specific | EMNIST Balanced / ByClass available from NIST, TensorFlow Datasets, Kaggle | Strong starting point for our own from-scratch embedded-friendly classifier. |
| CNN-SVM hybrid | Character classification | Community examples exist; no official Ref 7 repo found | Possible using CNN feature extractor + SVM classifier | No paper-specific checkpoints found | AHDB, AHCD, HACDB, IFN/ENIT for Arabic handwriting | Useful as a comparison model, but SVM classifier may complicate embedded training/deployment. |
| CNN-RNN / CNN-BiLSTM | Word/line recognition | Many community HTR implementations exist; no official Ref 9 repo found | Yes, but more complex | General HTR checkpoints exist in other projects, not paper-specific | IAM dataset commonly used; access/license needed | Better for full-word/line recognition than segmented-character classification, but heavier for embedded use. |
| CNN-LSTM-CTC | Offline HTR | Community code common; no official Ref 10/13 code found | Yes | Some community checkpoints exist depending on repo | IAM and language-specific datasets | Strong conceptual fit if character segmentation is too fragile, but may violate the proposal's planned segmentation/first-principles direction unless justified. |
| TrOCR | OCR / HTR recognition | Official Microsoft/unilm code and Hugging Face support | Yes | Yes: Microsoft checkpoints including `trocr-base-handwritten`; IAM handwritten, SROIE printed, scene-text variants | IAM, SROIE, scene-text benchmark datasets | Excellent benchmark/reference, but too large and too off-the-shelf for final proposal core unless used only for comparison. |
| Graves handwriting synthesis RNN | Stroke-level handwriting generation | Community TensorFlow implementations exist | Yes in community repos | Some repos include pretrained checkpoints | Original online handwriting-style data; repo-specific prepared data | Very relevant to trajectory generation, but we must be careful about off-the-shelf model use. |
| Sketch-RNN | Stroke/vector sketch generation | Official Magenta code | Yes | Yes: around 100 pretrained QuickDraw sketch models in Magenta JS | QuickDraw, Kanji, Omniglot, custom stroke datasets | Useful for stroke representation ideas, but not a text handwriting recognizer. |
| One-DM | One-shot handwriting text image generation | Official PyTorch code | Yes: `train.py`, `train_finetune.py`, `test.py` | Yes: One-DM, OCR model, ResNet18 weights linked | English datasets linked; paper reports English/Chinese/Japanese experiments | Excellent handwriting-style generation reference, but image-generation diffusion is likely too heavy/off-the-shelf for the embedded final system. |
| Robotic kinematic ML gender/legibility models | Writer/quality analysis | No official code found | Not found | Not found | BiosecurID, PaHaW, study-specific kinematic datasets | Useful for feature ideas and style/legibility metrics, less directly useful for writing reproduction. |

## Sources Checked

- `Research/ReferenceLinks.txt`
- `Research/Handwritten_Text_Classification_Based_on_Convoluti.pdf`
- NIST EMNIST page: https://www.nist.gov/itl/products-and-services/emnist-dataset
- TensorFlow Datasets EMNIST: https://www.tensorflow.org/datasets/catalog/emnist
- Cost-Effective Robotic Handwriting System with AI Integration: https://arxiv.org/abs/2501.06783
- One-DM paper/repo: https://arxiv.org/abs/2409.04004 and https://github.com/dailenson/One-DM
- Sketch-RNN / Magenta: https://github.com/magenta/magenta/blob/main/magenta/models/sketch_rnn/README.md and https://magenta.github.io/magenta-js/sketch/
- TrOCR: https://github.com/microsoft/unilm/blob/master/trocr/README.md and https://huggingface.co/docs/transformers/en/model_doc/trocr
- Handwriting synthesis community implementation: https://github.com/sjvasquez/handwriting-synthesis
- A Novel Handwritten Digit Classification System Based on CNN Approach: https://www.mdpi.com/1424-8220/21/18/6273
- Classification of Non-native Handwritten Characters Using CNN: https://arxiv.org/abs/2406.04511
- A Study on a Hybrid CNN-RNN Model for Handwritten Recognition Based on Deep Learning: https://www.scitepress.org/Papers/2023/128010/128010.pdf
- Handwriting-Based Gender Classification Using Robotic and Machine Learning Models: https://link.springer.com/article/10.1007/s12559-025-10478-2
- Intelligent Handwritten Recognition Using Hybrid CNN Architectures Based-SVM Classifier with Dropout: https://doi.org/10.1016/j.jksuci.2021.01.012

## Follow-Up Needed

- Download/extract refs 8, 10, 12, 14, 16, and 17 directly into local notes so their baseline lists are not based only on page snippets.
- For each candidate model we actually consider using, check:
  - embedded feasibility,
  - whether it violates the proposal's "library functions only for hardware interfacing" constraint,
  - whether it can be implemented from first principles,
  - how it will be evaluated against the 95% classification requirement,
  - and whether it can run within `2 + x*0.025` seconds on the selected single-board computer.
