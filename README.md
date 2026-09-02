# Voice-Following Teleprompter
### A Learned Advance-Decision Policy for Real-Time Speech-to-Script Alignment

> **AI Project — Victoria University of Wellington**

---

## Overview

Voice-following teleprompters advance an on-screen script by tracking the presenter's speech in real time. Current systems rely on **fixed-threshold lexical matching**, which breaks down when presenters paraphrase or skip sections — both of which happen naturally in any live presentation.

This project replaces the fixed-threshold policy with a **trained binary classifier** that decides, at each moment, whether to advance the script or hold — combining lexical and semantic similarity signals to handle natural speech variation.

A key design constraint: **advancing too early is catastrophic** (the audience sees the wrong text), while lagging behind is just annoying. The classifier is trained with asymmetric class weighting to reflect this.

---

## Research Question

> Can a learned advance-decision policy — combining lexical and semantic similarity — outperform fixed-threshold lexical matching, particularly under paraphrasing and section skips?

---

## System Pipeline

```
Microphone → Streaming ASR → Similarity Computation → Advance Classifier → Display
```

| Component | Tool | Role |
|---|---|---|
| Streaming ASR | `faster-whisper` / Vosk | Transcribes speech in real time (~0.5–1 s windows) |
| Semantic similarity | `sentence-transformers` (all-MiniLM-L6-v2) | Captures meaning-level similarity under paraphrase |
| Lexical similarity | `RapidFuzz` | Token-sort ratio between transcript and script chunk |
| Advance classifier | `scikit-learn` | Trained advance/hold decision model |
| Front-end | WebSocket (Python) | Displays the script live |

All inference runs on **CPU — no GPU required**.

---

## Classifier Features

The classifier is evaluated at every ASR update window and uses the following features:

1. Lexical similarity between transcript window and current chunk (RapidFuzz token-sort ratio)
2. Semantic cosine similarity between transcript and current chunk
3. Semantic cosine similarity between transcript and **next** chunk
4. Difference between (3) and (2) — positive values signal readiness to advance
5. Time elapsed on the current chunk
6. Proportion of current chunk tokens seen in the transcript so far
7. Rolling mean of lexical similarity over the last 3 windows
8. ASR confidence score

Candidate models: **logistic regression** and **small random forest**.  
Class weighting: false advances penalised **3× more** than false holds.

---

## Baselines

| Method | Description |
|---|---|
| **B1 — Fixed threshold** | Advance when RapidFuzz score > θ (tuned on validation split). Current industry standard. |
| **B2 — Hybrid rule** | Use lexical matching; fall back to semantic similarity when lexical score < secondary threshold. Hand-crafted, not learned. |
| **B3 — Learned classifier** | Trained advance/hold model combining all features above. Core contribution. |

---

## Dataset

| Parameter | Value |
|---|---|
| Scripts | 3 (600–800 words each, segmented into 40–60 chunks) |
| Conditions | 3 per script: literal reading · partial paraphrase · section skips |
| Speakers | 2 (non-native English speaker + native English volunteer) |
| Total recordings | 18 (~90 minutes of audio) |
| Positive instances | ~900 chunk transitions |
| Total decision instances | Several thousand (one per ASR window) |

**Ground truth labelling:** semi-automatic forced alignment + manual verification.  
**Dataset licence:** Creative Commons (to be released publicly — first reproducible evaluation resource for this task).

---

## Evaluation Metrics

- **Chunk-position accuracy** — proportion of windows where the displayed chunk matches ground truth
- **False-advance rate** — weighted 3× to reflect asymmetric error cost
- **Advance latency** — seconds between the correct moment and the actual advance
- **F1 on the advance class**
- **Paraphrase-tolerance trade-off curve** — how tolerance to paraphrase affects false-advance rate across methods

---

## Expected Outcomes

- **(a)** Working real-time prototype
- **(b)** First public annotated evaluation dataset with ground-truth alignment labels
- **(c)** Trained classifier + quantitative comparison of all three methods
- **(d)** Empirical findings on paraphrase tolerance limits per method and when semantic matching justifies its latency cost
- **(e)** Design conclusions on handling asymmetric error costs in real-time alignment systems

---

## Project Timeline

| Weeks | Milestone | Deliverable |
|---|---|---|
| 1–2 | Script writing, recording setup, B1 implementation | B1 system running |
| 3–4 | Data collection: 18 recordings, forced alignment, manual labelling | Annotated dataset |
| 5–6 | Feature extraction, classifier training, B2 implementation | Trained model + B2 |
| 7–8 | Evaluation, ablation experiments, error analysis | Results table |
| 9–10 | Real-time prototype integration and live demo | Working demo |
| 11–12 | Report writing and final submission | Final report |

---

## Stack

```
Python
├── faster-whisper          # Streaming ASR
├── sentence-transformers   # Semantic embeddings (all-MiniLM-L6-v2)
├── RapidFuzz               # Lexical similarity
├── scikit-learn            # Classifier training and evaluation
└── WebSocket (front-end)   # Live script display
```

---

## Extensions (time-permitting)

- Spanish recordings (the embedding model is multilingual)
- Noisy-audio conditions
- Additional speakers
- Semantic-only ablation

---

## Ethics

- All speakers provide informed consent for recordings used as training data
- Audio is processed locally — no data leaves the device
- Dataset released under Creative Commons licence

---

## References

[1] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP*, 2019.  
[3] M. Kan et al., "Slide-to-Speech Alignment for Lecture Video Indexing," *IEEE Trans. Multimedia*, 2007.

---

## Statement on Generative AI

Generative AI tools (Claude, Anthropic) were used to assist with drafting and structuring documentation. All technical decisions, experimental design, and project planning are the author's own.
