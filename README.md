# Dual-SSL Deepfake Audio Detection Pipeline

This repository contains a state-of-the-art Deepfake Audio Detection pipeline trained on the ASVspoof 2019 LA dataset. 

The model leverages a **Dual-SSL (Self-Supervised Learning)** architecture, combining the phonetic analysis capabilities of **Wav2Vec 2.0** with the acoustic and noise-robust capabilities of **WavLM**. By bridging these two pre-trained frontends with Bi-Directional Cross-Attention, the model achieves highly confident and decisive classification between *Bona Fide* (real) and *Spoofed* (fake) speech.

## 🏆 Benchmark Results (ASVspoof 2019 LA - Evaluation Set)
Tested on unseen, zero-day spoofing algorithms (A07–A19):
* **Global Accuracy:** 98.00%
* **Equal Error Rate (EER):** 2.00%
* **Macro F1-Score:** 0.9494

---

## 🧠 Model Architecture (`DualSSL_AASIST`)

The architecture fuses two distinct speech representations before passing them through an AASIST-inspired Graph Attention Pooling backend. 

```text
                                  +-----------------------------------+
                                  | Raw Audio Input (16 kHz, 4 sec)   |
                                  |      Tensor: [Batch, 64000]       |
                                  +-----------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
      +-----------------------------------+                 +-----------------------------------+
      |      Wav2Vec2 Base Frontend       |                 |       WavLM Base Plus Frontend    |
      |   Phonetic / Pronunciation Cues   |                 |    Acoustic / Denoising Cues      |
      |                                   |                 |                                   |
      | CNN Extractor: FROZEN             |                 | CNN Extractor: FROZEN             |
      | Transformer Layers 0-7: FROZEN    |                 | Transformer Layers 0-7: FROZEN    |
      | Transformer Layers 8-11: UNFROZEN |                 | Transformer Layers 8-11: UNFROZEN |
      +-----------------------------------+                 +-----------------------------------+
                         | [Batch, T, 768]                                     | [Batch, T, 768]
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                  +-----------------------------------+
                                  | Bi-Directional Cross-Attention    | 
                                  | 1. W2V2 Queries WavLM (8 heads)   |
                                  | 2. WavLM Queries W2V2 (8 heads)   |
                                  | 3. Concat [Batch, T, 1536]        |
                                  | 4. Linear Projection + LayerNorm  |
                                  +-----------------------------------+
                                                    |
                                                    v  [Batch, T, 768] 
                                  +-----------------------------------+
                                  |  AASIST Graph Attention Pooling   | 
                                  |   Linear(768 -> 256) -> Tanh      |
                                  |   Linear(256 -> 1) -> Softmax     |
                                  |   Temporal Weighted Sum           |
                                  +-----------------------------------+
                                                    |
                                                    v  [Batch, 768] 
                                  +-----------------------------------+
                                  |    Utterance Classifier Head      | 
                                  |   Linear(768 -> 256) -> GELU      |
                                  |   Linear(256 -> 64) -> GELU       |
                                  |   Linear(64 -> 1)                 |
                                  +-----------------------------------+
                                                    |
                                                    v
                                  +-----------------------------------+
                                  |   Output Logit (Real vs. Fake)    | 
                                  +-----------------------------------+



📂 Detailed File Overview & Workflow
This repository has been strictly modularized into 5 essential Python scripts to ensure clean, readable, and maintainable code.

1. dataset.py (Data Ingestion & Audio Processing)
Purpose: Handles all interactions with the raw .flac audio files and protocol text files.

parse_asvspoof_protocol(): Reads the ASVspoof text files, extracts the file names, and parses the ground truth labels (converting "bonafide" to 0.0 and "spoof" to 1.0).

SpoofDataset Class: A custom PyTorch Dataset. It uses torchaudio to load the waveforms. Crucially, it normalizes all audio lengths: if a clip is shorter than 4 seconds, it pads it with silence (zeros); if it is longer, it truncates it to exactly 64,000 samples (16 kHz * 4 seconds). This guarantees a uniform tensor shape [Batch, 64000] for the model.

2. model.py (Neural Network Architecture)
Purpose: Defines the DualSSL_AASIST neural network.

BiDirectionalCrossAttention: A custom module utilizing nn.MultiheadAttention to allow Wav2Vec2 and WavLM feature maps to query one another, fusing their distinct phonetic and acoustic representations.

GraphAttentionPooling: Compresses the resulting 3D sequence tensor [Batch, Time, Features] into a 2D utterance-level vector [Batch, Features] using learnable attention weights instead of simple average pooling.

DualSSL_AASIST: The main wrapper class. It instantiates the HuggingFace pre-trained models (facebook/wav2vec2-base and microsoft/wavlm-base-plus), disables gradient computation for the CNN feature extractors, enables gradient checkpointing to save VRAM, and chains the cross-attention and classifier heads together.

3. metrics.py (Evaluation Mathematics)
Purpose: Houses all mathematical functions for model evaluation.

compute_eer(): Leverages sklearn.metrics.roc_curve to calculate the Equal Error Rate. It dynamically finds the exact point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR) and returns both the EER and the mathematically optimal decision boundary (threshold).

compute_accuracy() & compute_macro_f1(): Applies the optimal threshold to the model's continuous scores to compute hard binary classification metrics.

4. train.py (The 2-Stage Training Loop)
Purpose: Manages the optimization process with a memory-safe, two-phase approach.

Stage 1 (Warmup): Freezes both massive transformer backbones entirely. It trains only the randomly initialized Cross-Attention block and Classifier head for 3 epochs at lr=1e-4 to prevent catastrophic gradient explosions.

Stage 2 (Fine-tuning): Unfreezes the top 4 layers (Layers 8-11) of both Wav2Vec2 and WavLM. It trains the network for 10 epochs using AdamW, CosineAnnealingLR, and Automatic Mixed Precision (AMP/FP16) to scale the loss safely. It automatically tracks the Dev Set EER and saves the best weights as dual_ssl_aasist_best.pt.

5. evaluate_dual_ssl.py (Benchmarking & Export)
Purpose: Runs the final validation on unseen data and exports insights.

Raw Logit Processing: Bypasses standard torch.sigmoid() conversion to prevent Float32 precision limits from artificially inflating the EER due to model overconfidence.

Outputs: Automatically plots the Detection Error Tradeoff (DET) curve (det_curve_dual_ssl.png) and generates a granular file-by-file error report (evaluation_results_dual_ssl.csv) for false positive/negative analysis.

🗄️ Dataset Setup & Preparation
To run this pipeline, you must prepare the ASVspoof 2019 Logical Access (LA) dataset. This dataset evaluates synthetic speech (Text-to-Speech and Voice Conversion) against real human speech.

Step 1: Download the Data
The dataset can be obtained from the official ASVspoof repository or Edinburgh DataShare. You need the LA (Logical Access) subset, which is approximately 24 GB of .flac audio files.

Step 2: Establish the Directory Structure
Extract the downloaded archives and arrange them exactly as follows on your local machine or server. The pipeline hardcodes this directory logic:

Plaintext
/home/guest/Desktop/DeepFake_Dataset/ALL Deepfake Data/AsvSpoof2019_LA/LA/
│
├── ASVspoof2019_LA_train/
│   └── flac/                  <-- Contains 25,380 .flac training files
│
├── ASVspoof2019_LA_dev/
│   └── flac/                  <-- Contains 24,844 .flac validation files
│
├── ASVspoof2019_LA_eval/
│   └── flac/                  <-- Contains 71,237 .flac evaluation files
│
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
Step 3: Understanding the Protocols
The .txt files in the protocols folder map the audio files to their ground truth labels.

A standard line looks like this: LA_0079 LA_T_1138215 - - bonafide

The dataset.py script reads column 2 (the filename: LA_T_1138215) and column 5 (the label: bonafide or spoof).

Note: You do not need to alter these text files. The data loader handles string parsing automatically.

(If your dataset is stored in a different location, open train.py and evaluate_dual_ssl.py and update the base_dir string variable).

⚙️ Installation & Prerequisites
1. Clone the repository:

Bash
git clone <your-repository-url>
cd <repository-name>
2. Install dependencies:
The code requires Python 3.8+ and a CUDA-enabled GPU.

Bash
pip install torch torchaudio transformers numpy matplotlib tqdm scikit-learn
🚀 Usage
Training the Model
To initiate the 2-stage training process, run:

Bash
python train.py
The script uses a batch_size of 4 to accommodate two full SSL models in VRAM. If you encounter CUDA Out-of-Memory (OOM) errors, drop this to 2.

The optimal checkpoint (based on lowest Dev Set EER) will be saved automatically as dual_ssl_aasist_best.pt. (Note: Checkpoints are git-ignored due to size limits).

Evaluating the Model
Once training is complete, test the model on the unseen evaluation set:

Bash
python evaluate_dual_ssl.pyc
This script will output:

Terminal summary of EER, Accuracy, and F1.

evaluation_results_dual_ssl.csv: A file-by-file breakdown of false positives and false negatives.

det_curve_dual_ssl.png: The Detection Error Tradeoff (DET) curve plot.

🔜 Future Work
With the baseline established on laboratory-clean ASVspoof data, the next phase of this project will integrate "In-The-Wild" datasets (ADD 2022/2023, LAV-DF) using PyTorch ConcatDataset balancing to enforce robustness against MP3 compression, social media background noise, and real-world acoustic masking.
