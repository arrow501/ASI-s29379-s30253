# data_processing — przygotowanie danych w Sprincie 2

Za zadanie #11 odpowiada Anton. Fabryka łączy `validate_data`, `normalize_data`
i `filter_data`; wejście CSV jest zdefiniowane w `conf/base/catalog.yml`
i nadpisane dla próbki w `conf/ci/catalog.yml`.

Wejście: `raw_data` i `params:project`. Wyjście: `prepared_data`, które registry
przekazuje bezpośrednio do `data_science`.

Kolejność: walidacja kontraktu → input normalization → training-only filtering.
Korzystamy z istniejących helperów w `asi_project.data`. Wynik zachowuje target,
indeks i kolejność rekordów; nie zawiera dopasowanego preprocessingu.
Może zostać w pamięci, zapis pod `data/02_intermediate/` jest opcjonalny.

[Dokładny kontrakt, wpisy catalogu i komendy odbioru](../../../../docs/notes/sprint-2-handoff.md).
