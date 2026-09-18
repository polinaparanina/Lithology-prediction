### 12 классов

| Модель | Выборка | Accuracy | Macro F1 | Weighted F1 |
|--------|---------|----------|----------|-------------|
| SGD baseline | val | 0.72 | 0.41 | 0.73 |
| SGD baseline | test | 0.71 | 0.47 | 0.72 |
| RF (Optuna) | val | 0.77 | 0.46 | 0.74 |
| RF (Optuna) | test | 0.76 | 0.46 | 0.73 |
| SGD baseline + DAE baseline | val | 0.72 | 0.41 | 0.73 |
| SGD baseline + DAE baseline | test | 0.70 | 0.45 | 0.71 |
| RF + DAE baseline | val | 0.77 | 0.48 | 0.74 |
| RF + DAE baseline | test | 0.76 | 0.49 | 0.73 |
| SGD baseline + DAE modified | val | 0.72 | 0.41 | 0.73 |
| SGD baseline + DAE modified | val | 0.70 | 0.44 | 0.70 |
| RF + DAE modified | val | 0.76 | 0.47 | 0.74 |
| RF + DAE modified | val | 0.76 | 0.47 | 0.72 |

### 5 классов (grouped)

| Модель | Выборка | Accuracy | Macro F1 | Weighted F1 |
|--------|---------|----------|----------|-------------|
| SGD baseline grouped | val | 0.81 | 0.53 | 0.82 |
| SGD baseline grouped | test | 0.84 | 0.63 | 0.84 |
| RF (Optuna) + CascadeImputer | val | 0.85 | 0.74 | 0.84 |
| RF (Optuna) + CascadeImputer | test | 0.85 | 0.75 | 0.84 |
| RF (Optuna) + HyperImpute1 | val | 0.86 | 0.78 | 0.85 |
| RF (Optuna) + HyperImpute1 | test | 0.86 | 0.82 | 0.86 |
| RF (Optuna) + HyperImpute2 | val | 0.86 | 0.78 | 0.85 |
| RF (Optuna) + HyperImpute2 | test | 0.85 | 0.80 | 0.85 |



### Выводы:
Наиболее точный и сбалансированный результат показала модель `RF (Optuna) + HyperImpute` (5 grouped classes). 

Группировка классов – единственный фактор, который стабильно и воспроизводимо улучшает метрики.

Использование DAE не дало существенного прироста качества. 

HyperImpute (missforest) на порядок медленнее в инференсе, чем готовые CatBoost-модели CascadeImputer. 

### Предложения по улучшению:
Рассмотреть возможность использования SMOTE (или аналогов)

Оптимизировать Optuna по macro F1, а не по weighted F1

Увеличить n_trials для grouped RF

