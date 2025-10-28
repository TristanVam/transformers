# Hybrid PatchTST Crypto Forecasting Pipeline

Ce projet implémente une architecture Transformer robuste pour prévoir les variations de prix ΔP (ou log-return) multi-horizon sur des données crypto Binance. Le système combine :

- **PatchTST amélioré** avec token `[CLS]` appris, embedding de symbole et régulation de régime (volatilité/ATR simplifié) pour stabiliser la tête variationnelle.
- **Tête variationnelle (μ, σ)** avec clamp et régularisation L2 sur `log_sigma` pour produire des intervalles calibrés.
- **Fusion de sentiment FinBERT** via cross-attention avec gating sigmoïde, dropout et pooling par token `[CLS]`.
- **Augmentation brownienne corrélée** (drift/sigma estimés sur la dernière portion de la fenêtre, bruit corrélé prix/volume) appliquée uniquement aux splits d’entraînement.
- **Validation walk-forward** et ensembling SGDR pour simuler un déploiement réel et lisser les prédictions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch doit être installé avec l’extension GPU appropriée si nécessaire. TensorBoard est utilisé pour les logs.

## Configuration

Tous les hyperparamètres sont centralisés dans `src/training/config.py` et peuvent être surchargés via un JSON (exemple : `configs/experiment_example.json`). Les champs couvrent :

- Paramètres data (multi-symboles, taille de fenêtre, horizons multiples, demi-vie du sentiment)
- Augmentation brownienne (ρ, nombre de trajectoires, limite par fenêtre)
- Optimiseur (lr, weight decay, betas, eps)
- Entraînement (batch size, scheduler cosine + warmup, mixed precision, clipping, seed, patience early stopping, workers, répertoires de sortie/logs)

## Lancement d’un entraînement

```bash
python scripts/train.py --config configs/experiment_example.json --log-level INFO
```

Le pipeline :

1. Télécharge les chandeliers horaires depuis Binance (avec cache `.npy`, retry/backoff, validation UTC).
2. Prépare un encodeur FinBERT et une fonction de lookup de sentiment (à adapter pour votre source de tweets).
3. Construit les jeux de données avec split temporel strict, moyenne exponentielle du sentiment et augmentation brownienne contrôlée.
4. Entraîne le modèle avec :
   - **Mixed precision** (`torch.cuda.amp`) + `GradScaler`
   - **Clipping** des gradients (`max_norm = 1.0`)
   - **Scheduler CosineAnnealingLR avec warmup**
   - **Early stopping** sur la perte de validation et sauvegarde du meilleur checkpoint
   - **Logging TensorBoard** (loss, métriques, attention, PnL, Sharpe, coverage, etc.)
   - **Sauvegarde d’état complet** (modèle, optimiseur, scheduler, scaler, epoch)
5. Calcule une validation **walk-forward** multi-fold, agrège les métriques (MAE, RMSE, NLL, coverage, hit ratio, PnL simulé, Sharpe) et moyenne les snapshots récents pour un ensembling SGDR.
6. Sauvegarde les cartes d’attention (`checkpoints/attention/attention_epoch_*.pt`) et le checkpoint final.

Les logs TensorBoard peuvent être visualisés via :

```bash
tensorboard --logdir runs
```

## Visualisation & Explicabilité

- **Cartes d’attention** : le pipeline enregistre les poids de cross-attention sentiment/temps. Utilisez `python -m src.utils.attention_viz path/to/attention_epoch_X.pt --head 0 --output attention.png` pour générer une heatmap.
- **Analyse des têtes** : les histogrammes TensorBoard exposent la distribution des poids par tête, permettant de contrôler l’entropie et les heads dominantes.

## Structure du dépôt

```
src/
├── data/
│   ├── augmentation.py      # Augmentation brownienne corrélée et contrôlée
│   ├── binance.py           # Client REST Binance avec cache et retry
│   └── dataset.py           # Dataset multi-symboles, sentiment pondéré, multi-horizon
├── models/
│   ├── patchtst.py          # PatchTST + fusion sentiment + gating de régime
│   ├── sentiment.py         # Cross-attention FinBERT avec gating sigmoïde
│   └── variational.py       # Tête variationnelle (μ, σ) multi-horizon
├── training/
│   ├── config.py            # Dataclasses & sérialisation JSON
│   └── pipeline.py          # Pipeline complet (AMP, scheduler, early stopping, walk-forward)
└── utils/
    ├── attention_viz.py     # Script de visualisation d’attention
    └── sentiment.py         # Encodage FinBERT et helpers

scripts/
└── train.py                 # Entrée CLI
```

## Prochaines étapes

- Connecter la fonction de sentiment à une source temps réel (Twitter, actualités) et enrichir les features (FinBERT + volume social).
- Brancher la sortie du modèle à une plateforme de trading via API en exploitant les intervalles de confiance pour dimensionner les positions.

> **Disclaimer** : Ce code est fourni à des fins de recherche et d’expérimentation. Le trading algorithmique comporte des risques significatifs.
