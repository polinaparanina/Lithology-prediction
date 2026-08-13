### 12 raw классов

| Модель | Выборка | Accuracy | Macro F1 | Weighted F1 |
|--------|---------|----------|----------|-------------|
| SGD baseline | val | 0.74 | 0.42 | 0.74 |
| SGD baseline | test | 0.74 | 0.50 | 0.73 |
| RF (Optuna) | val | 0.75 | 0.47 | 0.75 |
| RF (Optuna) | test | 0.73 | 0.44 | 0.72 |
| RF + DAE baseline | val | 0.76 | 0.50 | 0.76 |
| RF + DAE modified | val | 0.79 | 0.56 | 0.75 |

### 5 grouped классов

| Модель | Выборка | Accuracy | Macro F1 | Weighted F1 |
|--------|---------|----------|----------|-------------|
| SGD baseline grouped | val | 0.82 | 0.56 | 0.82 |
| SGD baseline grouped | test | 0.85 | 0.67 | 0.84 |
| RF (Optuna) + CascadeImputer | val | 0.85 | 0.75 | 0.84 |
| RF (Optuna) + CascadeImputer | test | 0.84 | 0.75 | 0.83 |
| RF (Optuna) + HyperImpute | val | 0.86 | 0.77 | 0.85 |
| RF (Optuna) + HyperImpute | test | — | — | — |