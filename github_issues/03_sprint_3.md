# Sprint 3 — Eksperymenty, wybór modelu i artefakt

> Treść generowana z `docs/sprints/SPRINT_3.md` przez `python scripts/build_issues.py`. Zmiany wprowadzaj w pliku sprintu.

**Daty:** 12.11.2026 – 26.11.2026

## Cel

Zespół porównuje eksperymenty, wybiera model produkcyjny i zapisuje go jako pakiet,
który można wykorzystać bez ponownego treningu.

## Definition of Done

- [ ] Baseline jest zapisany jako run W&B.
- [ ] Runy W&B zawierają parametry, metryki, Git SHA i identyfikator wersji danych.
- [ ] AutoGluon działa jako część pipeline'u.
- [ ] Wykonano co najmniej dwie sensowne konfiguracje AutoGluona z limitem czasu.
- [ ] Przygotowano tabelę porównującą modele.
- [ ] Porównanie obejmuje główną metrykę, czas predykcji, rozmiar artefaktu i pamięć przy ładowaniu.
- [ ] Zespół wybrał model produkcyjny.
- [ ] Pakiet modelu zawiera model i `metadata.json`.
- [ ] Model jest zapisywany i wczytywany wspólnymi funkcjami.
- [ ] Pakiet modelu jest zapisany w W&B jako Artifact.
- [ ] Artefakt produkcyjny ma alias `production`.
- [ ] Smoke test pobiera `production`, wczytuje model i wykonuje predykcję.
- [ ] `docs/model_card.md` jest uzupełniony.
- [ ] ADR-2 jest napisany i zmergowany.
- [ ] CI na `main` jest zielone.
- [ ] Każda osoba w zespole ma własny obszar pracy i zrobiła merytoryczne review pracy innej osoby.

## Podział pracy (uzupełnia zespół na początku sprintu)

| Element sprintu | Owner | Recenzent |
| --- | --- | --- |
| S3-1 — tracking W&B i eksperymenty AutoGluona w Kedro | Arrow (arrow501, s29379) | Anton (sanderJa, s30253) |
| S3-2 — pomiary, porównanie modeli, ADR-2 i Model Card | Arrow (arrow501, s29379) | Anton (sanderJa, s30253) |
| S3-3 — wspólny pakiet modelu, metadane i W&B Artifact | Anton (sanderJa, s30253) | Arrow (arrow501, s29379) |
| S3-4 — CI, niezależny smoke test i przekazanie do Sprintu 4 | Anton (sanderJa, s30253) | Arrow (arrow501, s29379) |

Integrator sprintu: Arrow (arrow501, s29379). Wybór modelu produkcyjnego zatwierdzają
obie osoby. Każdy przygotowuje testy własnych zmian i robi merytoryczne review drugiej osoby.

Podział zakłada orientacyjnie 55–60% pozostałego nakładu po stronie Arrow, bez wliczania
wykonanych wizualizacji marimo. Marimo pozostaje opcjonalną eksploracją; wymagane
eksperymenty, raporty i odbiór działają bez niego.

Przed implementacją obie osoby uzgadniają dane przekazywane do W&B, wynik eksperymentu,
interfejs zapisu/odczytu pakietu i schemat predykcji. Arrow zapisuje uzgodnienia
w dokumentacji pipeline'u, Anton uzupełnia kontrakt pakietu w `src/asi_project/README.md`.
S3-1 i S3-3 mogą być rozwijane równolegle; Anton używa baseline'u do pracy nad pakietem
i smoke testem. S3-2 korzysta z runów S3-1 i wspólnych funkcji pakietu S3-3.
Odbiór S3-4 wymaga wybranego modelu i rzeczywistego artefaktu `production`.

Lokalne opisy zadań do publikacji:

- S3-1: `github_issues/tasks/s3-1-tracking-experiments.md`
- S3-2: `github_issues/tasks/s3-2-model-selection.md`
- S3-3: `github_issues/tasks/s3-3-model-package.md`
- S3-4: `github_issues/tasks/s3-4-integration-smoke.md`

## Pełny opis

Plik `docs/sprints/SPRINT_3.md` w repozytorium.
