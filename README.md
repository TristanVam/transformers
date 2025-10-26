# Hybrid Stochastic PatchTST for BTC/USDT ΔP Forecasting

Ce projet met en œuvre une architecture Transformer hybride capable de prévoir la
variation de prix horaire (ΔP(t+1)) du pair BTC/USDT. Le pipeline combine plusieurs
composants clés :

1. **Baseline PatchTST** – Embedding en patchs sur les séries OHLCV pour exploiter
   les dépendances temporelles à long terme.
2. **Brownian Data Augmentation** – Génération de trajectoires de type mouvement
   brownien géométrique pour exposer le modèle à des scénarios de volatilité variés.
3. **Variational Head** – Prédiction conjointe de la moyenne et de la volatilité
   (μ, σ) au lieu d’une valeur unique, permettant une meilleure calibration du risque.
4. **Fusion de Sentiment** – Cross-attention entre les représentations temporelles et
   des embeddings FinBERT extraits de tweets/flux d’actualité.

L’architecture est reliée à l’API publique de Binance afin de télécharger automatiquement
les chandeliers horaires BTC/USDT et de lancer l’entraînement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Remarque :** PyTorch et `transformers` (Hugging Face) sont requis. Adaptez
> l’installation de PyTorch (CPU/GPU) selon votre environnement.

## Entraînement

```bash
python scripts/train.py --log-level INFO
```

Vous pouvez fournir un fichier JSON pour personnaliser les hyperparamètres
(`--config path/to/config.json`). Les paramètres disponibles sont décrits dans
`src/training/config.py`.

Le script téléchargera automatiquement les données, entraînera le modèle et
sauvegardera le checkpoint dans `checkpoints/patchtst_hybrid.pt`.

## Structure du projet

```
src/
├── data/
│   ├── augmentation.py      # Brownian data augmentation
│   ├── binance.py           # Client REST Binance pour les chandeliers
│   └── dataset.py           # Dataset PatchTST avec sentiment et augmentation
├── models/
│   ├── patchtst.py          # Backbone PatchTST + fusion sentiment
│   ├── sentiment.py         # Cross-attention avec embeddings FinBERT
│   └── variational.py       # Tête variationnelle (μ, σ)
├── training/
│   ├── config.py            # Dataclasses de configuration
│   └── pipeline.py          # Pipeline d’entraînement complet
└── utils/
    └── sentiment.py         # Encodage FinBERT des tweets

scripts/
└── train.py                 # Point d’entrée CLI
```

## Tweets & FinBERT

Le module `src/utils/sentiment.py` encapsule FinBERT pour transformer des textes
financiers courts (tweets) en embeddings. Dans le pipeline fourni, un texte de
placeholder est utilisé ; il suffit de remplacer cette logique par une récupération
réelle des tweets via l’API de votre choix afin de bénéficier pleinement de la
fusion de sentiment.

## Licence

Projet fourni à titre éducatif. Utilisation sous votre propre responsabilité pour
le trading algorithmique.
