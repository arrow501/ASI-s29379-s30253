# Sprint 3: zamknięcie i przekazanie do S4

Stan na 26.11.2026, po merge #28 (`main`: `eb88556`). Wymagania S3 są spełnione.
Smoke online oraz zwykłe CI na Linuxie, Windowsie i macOS przeszły na `main`.

## Definition of Done

| Wymaganie S3 | Stan i dowód |
| --- | --- |
| Baseline i runy W&B: parametry, metryki, Git SHA, fingerprint | OK: [mapa runów](sprint-3-wandb-runs.json), [pełny przebieg online](sprint-3-online-integration.md) |
| AutoGluon w Kedro, co najmniej dwie konfiguracje z limitem czasu | OK: baseline + cztery konfiguracje, limit 120 s na fit; [konfiguracje i protokoły](sprint-3-local-experiments.md) |
| Tabela: jakość, czas predykcji, rozmiar i pamięć | OK: [porównanie](sprint-3-model-comparison.md), [pomiary pakietów](results/sprint-3-package-benchmark.md) |
| Wspólny wybór modelu, Model Card i scalony ADR-2 | OK: [Model Card](../model_card.md), [ADR-2](../adr/0002-wybor-modelu.md), review Antona w [#27](https://github.com/arrow501/ASI-s29379-s30253/pull/27) |
| Pakiet: model, metadata.json, wspólne funkcje zapisu i odczytu | OK: [kontrakt i komendy](../../src/asi_project/README.md#komendy-obsługi-pakietu), scalony [#25](https://github.com/arrow501/ASI-s29379-s30253/pull/25) |
| W&B Artifact z aliasem production | OK: [potwierdzenie publikacji v0](results/sprint-3-production.json) |
| Pobranie production, loader, pięć predykcji w nowym procesie bez fit | OK: [raport z main](results/sprint-3-smoke-main.json), [workflow online](https://github.com/arrow501/ASI-s29379-s30253/actions/runs/45860830536) |
| Instalacja od zera i niezależny odbiór | OK: [nowe środowisko conda](results/sprint-3-fresh-env-verification.json), [smoke online](results/sprint-3-smoke-fresh-env.json); Anton potwierdził własny odbiór instrukcji |
| CI na main, w tym lokalny zapis/odczyt bez sekretów | OK: [przebieg CI na trzech systemach](https://github.com/arrow501/ASI-s29379-s30253/actions/runs/45860830538) |
| Własne obszary pracy i wzajemne review | OK: Arrow #18/#19, Anton #20/#21; akceptacja Antona w #27 i Arrow w [#28](https://github.com/arrow501/ASI-s29379-s30253/pull/28) |

Odbiór Arrow wykonano na czystym checkoutcie `e9b3084`, Python 3.11.18 na Linuxie.
Środowisko utworzono z `environment.yml`; projekt zainstalowano przez
`python -m pip install --no-deps -e .`, a zależności przez:

```bash
python -m pip install -r requirements/preflight-s2.txt -r requirements/preflight-s3.txt -r requirements/benchmark-s3.txt -c constraints.txt
```

Pełny zestaw z `--run-autogluon-package`: **530 passed, bez pominięć**.
`pip check`, `check_env.py --autogluon`, pre-commit i `kedro run --env ci` przeszły.
Weryfikacja korzystała z istniejącego uwierzytelnienia W&B i cache pobierania;
bibliotek nie kopiowano z `asi-ml`. Raport Actions potwierdza dodatkowo wykonanie
smoke na scalonym `main`, w środowisku runnera.

## Model przekazany do Sprintu 4

- Pakiet: `wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/bank-marketing-model:production`.
- Zatwierdzona wersja: `bank-marketing-model:v0`, chronologiczny baseline.
- Próg z metadanych: `0.07166958577186548`; klasa pozytywna: `yes`.
- API używa `load_model_package` i `predict_model_package`; wejście bez targetu,
  wynik `prediction` / `probability_positive` w kolejności wejścia.
- Pobranie wymaga W&B online, predykcja działa lokalnie. Dostęp do konta jest
  tymczasowy; możliwe opóźnienia i błędy usługi opisuje [notatka W&B](sprint-3-wandb.md).
- Ograniczenia jakości, dryf klas i wcześniejsze oglądanie testu pozostają opisane
  w Model Card. Start S4 nie wymaga ponownego treningu ani zmiany wyboru modelu.
