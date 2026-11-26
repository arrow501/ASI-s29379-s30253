# Porównanie pakietów modeli — Sprint 3

Pomiar wspólnego loadera i predykcji z #20; rozmiar obejmuje cały pakiet. Predykcja zawiera normalizację wejścia, preprocessing i zapisany próg. RSS obejmuje proces wraz z bibliotekami.

## chronological — `20261123T232543-chronological-3d2c20c4`

| Model | F1 | Precision | Recall | Accuracy | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.5210 | 0.3955 | 0.7630 | 0.5569 | 2857 / 3330 / 677 / 2179 |
| gbm_small | 0.4923 | 0.3347 | 0.9300 | 0.3941 | 908 / 5279 / 200 / 2656 |
| gbm_large | 0.4817 | 0.3274 | 0.9107 | 0.3810 | 844 / 5343 / 255 / 2601 |
| catboost | 0.4800 | 0.3158 | 1.0000 | 0.3158 | 0 / 6187 / 0 / 2856 |
| tree_ensemble | 0.4800 | 0.3158 | 1.0000 | 0.3158 | 0 / 6187 / 0 / 2856 |

| Model | Batch [ms] | 1 rekord [ms] | Load [s] | Artefakt [MiB] | RSS przed [MiB] | RSS load / Δ [MiB] | RSS warm / Δ [MiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 23.72 | 2.83 | 0.017 | 0.006 | 175.2 | 180.8 / 5.7 | 185.8 / 10.6 |
| gbm_small | 17.33 | 4.82 | 0.153 | 0.136 | 175.0 | 200.7 / 25.7 | 213.4 / 38.5 |
| gbm_large | 17.68 | 4.82 | 0.152 | 0.241 | 175.0 | 200.8 / 25.8 | 212.9 / 37.8 |
| catboost | 20.39 | 6.32 | 0.154 | 0.092 | 174.8 | 200.6 / 25.8 | 237.0 / 62.2 |
| tree_ensemble | 92.66 | 68.17 | 0.150 | 116.930 | 175.0 | 200.8 / 25.8 | 401.0 / 226.0 |
