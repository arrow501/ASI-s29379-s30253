# Sprint 2 — Reprodukowalny pipeline

> Treść generowana z `docs/sprints/SPRINT_2.md` przez `python scripts/build_issues.py`. Zmiany wprowadzaj w pliku sprintu.

**Daty:** 29.10.2026 – 12.11.2026

## Cel

Zespół przenosi proces treningowy z notebooka do Kedro, tak aby cały proces od danych
do modelu i metryk można było odtworzyć jedną komendą.

## Definition of Done

- [ ] Kedro działa w przypiętej wersji z `requirements/preflight-s2.txt`.
- [ ] Pipeline `data_processing` wykonuje walidację, input normalization i training-only filtering.
- [ ] Pipeline `data_science` wykonuje split, trening i ewaluację.
- [ ] Pipeline korzysta z logiki przygotowanej w Sprincie 1 zamiast ją duplikować.
- [ ] Model preprocessing nadal znajduje się wewnątrz sklearn `Pipeline`.
- [ ] Parametry są przeniesione do konfiguracji.
- [ ] Data Catalog opisuje wejścia i wyjścia pipeline'u.
- [ ] `conf/ci/` pozwala uruchomić pipeline na próbce CI.
- [ ] Testy nodów przechodzą.
- [ ] `kedro run` na pełnych danych tworzy model i metryki.
- [ ] `kedro run --env ci` działa w CI.
- [ ] Notebook nie jest już wymagany do trenowania baseline'u.
- [ ] ADR-1 jest napisany i zmergowany.
- [ ] README zawiera sekcję „Kedro quickstart”.
- [ ] CI na `main` jest zielone.
- [ ] Każda osoba w zespole ma własny obszar pracy i zrobiła merytoryczne review pracy innej osoby.

## Podział pracy (uzupełnia zespół na początku sprintu)

| Element sprintu | Owner | Recenzent |
| --- | --- | --- |
| PR S2-1 — konfiguracja projektu Kedro, rejestracja pipeline'ów i ustalenie interfejsów | Arrow (arrow501, s29379) | Anton (sanderJa, s30253) |
| PR S2-2 — pipeline `data_processing`, catalog wejściowy dla pełnych danych i próbki CI oraz testy nodów | Anton (sanderJa, s30253) | Arrow (arrow501, s29379) |
| PR S2-3 — pipeline `data_science`, parametry treningu, zapis modelu i metryk oraz testy | Arrow (arrow501, s29379) | Anton (sanderJa, s30253) |
| PR S2-4 — integracja: uruchomienie w CI, ADR-1, quickstart w README, diagram architektury i smoke test na świeżym klonie | Anton (sanderJa, s30253) | Arrow (arrow501, s29379) |

Integrator sprintu: Anton (sanderJa, s30253) — jako owner PR S2-4 pilnuje integracji, Definition of Done i stanu `main` przed zajęciami 12.11.2026.

Ustalenia startowe i stan pre-flightu: [notatka z 29.10](../notes/sprint-2-start.md).

Każdy wiersz tabeli odpowiada jednemu PR. Najpierw scalany jest PR S2-1. Następnie
PR S2-2 i PR S2-3 mogą być rozwijane równolegle; końcowa weryfikacja i scalenie PR S2-3
wymagają zintegrowanego PR S2-2. PR S2-4 zamyka sprint po scaleniu obu pipeline'ów.
Arrow odpowiada za ustawienia projektu Kedro i rejestrację pipeline'ów; Anton za wpisy wejściowe
w catalogu, a Arrow za wpisy wyjściowe. Każdy przygotowuje testy swojego kodu i otrzymuje
merytoryczne review drugiej osoby.

W ramach PR S2-4 Anton wykonuje niezależny smoke test na świeżym klonie gałęzi PR:
uruchamia pipeline na pełnych danych i próbce CI, sprawdza zapis modelu i metryk oraz porównuje
wyniki ze Sprintem 1. Polecenia i wyniki zapisuje w opisie PR lub notatce sprintu,
a Arrow sprawdza je podczas review. Każdy wcześniejszy PR zawiera weryfikację własnych zmian.

Interfejs między pipeline'ami to `prepared_data`: DataFrame po walidacji, normalizacji
i filtrowaniu, nadal z targetem `y`, z zachowaną kolejnością wierszy. `data_science` wybiera
cechy z konfiguracji i wykonuje podział `time_ordered`. Imputacja, skalowanie i kodowanie
pozostają wewnątrz sklearn `Pipeline`, dopasowywanego wyłącznie na części treningowej.

W S2-1 trzeba ustalić sposób przekazania ustawień z `project.yaml` do Kedro: parametry nodów
pochodzą z konfiguracji Kedro, a reguły kontraktu nadal z `datasets.yaml`. Nie utrzymujemy dwóch
niezależnych zestawów aktywnych wartości splitu i list cech. Ścieżki wejść i wyjść należą do
Data Catalog. Sposób powiązania konfiguracji opisujemy w quickstarcie.

## Pełny opis

Plik `docs/sprints/SPRINT_2.md` w repozytorium.
