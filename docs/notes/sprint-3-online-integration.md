# Sprint 3 — pełny przebieg online

26.11.2026 sprawdzono rzeczywistą integrację Kedro, treningu, W&B i pakietów #20.
Wynik: **pięć runów i pięć pakietów zweryfikowanych przez API**.
[Dowód JSON](results/sprint-3-online-integration.json) zawiera linki, metryki,
wersje artefaktów oraz digesty pobranych pakietów.

## Źródło

- Czysty commit `52a0956`, z poprawionym loaderem #20 (`7bfd06f`); `git_dirty=false`.
- Nowy trening: `20261126T150209-chronological-b3ac4ba5`, pełne 45 211 rekordów.
- Trzy okna chronologiczne, refit 80%, wspólny test 9043 rekordów.
- Baseline i cztery konfiguracje AutoGluona; 120 s na fit, 8 CPU, bez GPU.
- W&B online w tymczasowym projekcie prowadzącej; kod zakończenia Kedro: 0.
  Pipeline wykonał wszystkie 9 nodów w około 50 s. Limit czasu jest górną granicą.

```bash
WANDB_MODE=online KEDRO_DISABLE_TELEMETRY=1 kedro run --env local --pipelines experiments_crossfold_test
```

To nowy przebieg integracyjny, bez zastępowania wyboru historycznego baseline’u
w [ADR-2](../adr/0002-wybor-modelu.md).

## Sprawdzenie W&B

| Model | Run W&B | F1 testu |
| --- | --- | ---: |
| `baseline` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/f269a5fb2abeb582b077) | 0.5210 |
| `catboost` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/d6c7497d4643149ea9ad) | 0.4800 |
| `gbm_large` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/5b631d3c0083bce827ab) | 0.4817 |
| `gbm_small` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/dc512208d916ebae3903) | 0.4923 |
| `tree_ensemble` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/3cc20283dade8ecd4d55) | 0.4800 |

Każdy run ma stan `finished`; metryki liczbowe zgadzają się z lokalnymi wynikami.
Sprawdzono SHA/dirty, fingerprint, identyfikator runu i kandydata oraz konfigurację
projektu. Sześć raportów każdego modelu pobrano z W&B i porównano bajt po bajcie
z lokalnymi plikami.

Każdy pakiet opublikowano jako artefakt typu `model`, w osobnej kolekcji
`s3-online-check-<run_id>-<candidate>`, wersja `v0`. Pełne `artifact_ref` są w JSON.
Pakiety pobrano przez funkcję #20; rozmiary i digesty odpowiadają źródłom.
Wszystkie mają alias `latest`; **nie nadawano ani nie zmieniano `production`**.

## Predykcja po pobraniu

Oryginalne katalogi treningu i lokalnych pakietów były na czas próby niedostępne.
Nowy proces, bez W&B i z zablokowanym `fit`, wczytał pięć pobranych pakietów.
Na 9043 rekordach zachował indeks, etykiety i prawdopodobieństwa
(`rtol=atol=1e-12`); macierze pomyłek i F1 odpowiadają raportom treningu.
Katalogi źródłowe przywrócono po próbie.

## Ostrzeżenia i zakres

Minimalny zestaw `autogluon.tabular[lightgbm,catboost]` nie instaluje PyTorcha.
Log zawiera komunikaty `Failed to import torch or check CUDA availability`
oraz informację o pominiętej kalibracji prawdopodobieństw. Backendów CPU
LightGBM, CatBoost, RF i XT nie zabrakło; zakończyły trening poprawnie.
Próg F1 był dobierany osobno przez kod eksperymentów, zgodnie z konfiguracją.
Nie traktujemy pominiętej kalibracji jako wykonanej ani jako awarii W&B.

Próba używała tego samego środowiska conda, w którym wykonano trening.
Odbiór #21 wykonano później na pakiecie `production` w nowym środowisku i w Actions:
[wyniki końcowe S3](sprint-3-closeout.md).
Dostęp jest [tymczasowy](sprint-3-wandb.md); możliwe są opóźnienia i błędy usługi.
