# Projet initialisé

Ce dépôt contient maintenant un pipeline Transformer multi-marchés
destiné à la prévision de la prochaine bougie (`horizon = 1`). Le
fichier `market_specific_transformers.py` propose trois variantes
spécialisées :

| Marché  | Objectif principal | Particularités |
|---------|-------------------|----------------|
| NASDAQ  | Actions US         | Fenêtre 90 pas, encodeur 4 couches, indicateurs momentum courts/moyens |
| OR      | Futures/CFD or     | Fenêtre 120 pas, encodeur 3 couches, ATR longue et momentum lissé |
| BTC     | Spot crypto        | Fenêtre 72 pas, encodeur 5 couches, dropout renforcé |

## Utilisation rapide

```bash
python market_specific_transformers.py --market nasdaq --mode train --csv path/to/candles.csv
```

Modes disponibles :

- `train` : entraîne le modèle du marché sélectionné et sauvegarde les
  poids ainsi que la liste des features.
- `evaluate` ou `backtest` : recharge un modèle entraîné et lance le
  backtest interne (Sharpe, max drawdown, etc.).
- `hpsearch` : lance une recherche hyperparamétrique Optuna autour de la
  configuration du marché.

Les CSV attendus doivent contenir les colonnes suivantes :
`timestamp`, `open`, `high`, `low`, `close`, `volume`.
