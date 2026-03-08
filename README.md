# Unified Adaptive Framework for Multimodal Data Fusion
## with Static Modality Reweighting (Phase 1)

### Dataset
Hateful Memes — HuggingFace: emily49/hateful-memes

### Architecture
CLIP Encoders → Cross-Modal Attention → Static Gating → Classifier

### Phase 1 Results
| Model | AUROC |
|---|---|
| Text Only | 0.6314 |
| Image Only | 0.6326 |
| Concat Fusion | 0.6890 |
| Cross-Attn (no reweighting) | 0.7054 |
| Cross-Attn + Static Reweighting | 0.7131 |

### Run
pip install -r requirements.txt
python3 src/train.py
python3 src/run_ablation.py
