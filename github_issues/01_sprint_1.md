# Sprint 1 — Problem, dane i poprawny baseline

> Treść generowana z `docs/sprints/SPRINT_1.md` przez `python scripts/build_issues.py`. Zmiany wprowadzaj w pliku sprintu.

**Daty:** 15.10.2026 – 29.10.2026

## Cel

Zespół wybiera dataset, ustala zasady korzystania z danych i przygotowuje poprawny baseline
w sklearn `Pipeline` z działającymi testami i CI.

## Definition of Done

- [x] Dataset jest wybrany i wpisany do `project.yaml`.
- [ ] Karta projektu w `README.md` jest kompletna.
- [ ] Pełne dane są pobierane skryptem i nie znajdują się w repozytorium.
- [ ] `data/01_raw/sample.csv` znajduje się w repozytorium i zawiera obie klasy.
- [x] `project.yaml` zawiera decyzje zespołu dotyczące typów cech i splitu.
- [x] `validate()` przechodzi dla danych projektu.
- [x] `normalize_input()` jest zaimplementowane i przechodzi testy inference-safe.
- [x] `filter_training_rows()` jest zaimplementowane, jeśli dataset tego wymaga.
- [x] `notebooks/01_eda.ipynb` zawiera EDA oraz zapisane decyzje.
- [ ] `DummyClassifier` i `LogisticRegression` są porównane.
- [ ] `LogisticRegression` działa wewnątrz sklearn `Pipeline`.
- [ ] `predict_proba()` działa.
- [ ] `metrics.json` pochodzi z pełnych danych.
- [ ] `docs/architecture.md` zawiera diagram v0.
- [ ] CI na `main` jest zielone.
- [ ] Żaden wymagany test z `tests/test_project.py` nie jest pominięty.
- [ ] Każda osoba w zespole ma własne zadanie i zrobiła merytoryczne review pracy innej osoby.

## Podział pracy (uzupełnia zespół na początku sprintu)

| Element sprintu | Owner | Recenzent |
| --- | --- | --- |
| Dataset, karta projektu i próbka CI | Arrow (s29379) | Anton (s30253) |
| EDA i decyzje o typach cech | Anton (s30253) | Arrow (s29379) |
| Input normalization i training-only filtering | Anton (s30253) | Arrow (s29379) |
| sklearn Pipeline, baseline i metryki | Arrow (s29379) | Anton (s30253) |
| Diagram architektury v0, CI i integracja | Arrow (s29379) | Anton (s30253) |

Integrator sprintu: Arrow (s29379) — koordynacja Definition of Done, smoke testu i demo.

## Pełny opis

Plik `docs/sprints/SPRINT_1.md` w repozytorium.
