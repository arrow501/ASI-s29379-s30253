# Model Card — Bank Marketing

Stan: **kandydat do pakowania, niezatwierdzony `production`** (24.11.2026).
Opis dotyczy konkretnego modelu wskazanego poniżej; po wspólnej decyzji z Antonem
trzeba dopisać wersję paczki i link W&B. [ADR-2](adr/0002-wybor-modelu.md) jest propozycją.

## Model i pochodzenie

- Kandydat: `baseline`, run `20261123T232543-chronological-3d2c20c4`, Arrow.
- Regresja logistyczna (`max_iter=1000`, seed 42) w sklearn Pipeline.
  Liczby: imputacja medianą i StandardScaler; kategorie: imputacja najczęstszą
  wartością i OneHotEncoder z obsługą nieznanych kategorii. Wszystko dopasowane na train.
- Trening: trzy rozszerzające się okna w pierwszych 80%, wybór według średniego F1,
  następnie refit na całych 36 168 rekordach development.
- Decyzja: `P(y=1) >= 0.07166958577186548`; próg dobrany na predykcjach forward-fold,
  bez końcowego testu. Nie używać domyślnego `predict()` z progiem 0,5.
- SHA zapisany przez trening: `c66e7f81e3e8bb1aa3630e165b7f24c768250508`,
  `git_dirty=true`; sam SHA nie odtwarza całego ówczesnego kodu.
- Wersje treningu: Python 3.11.17, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3.
  Model jest lokalnym pickle z eksperymentów; paczka #20, jej wersja i W&B są jeszcze
  niedostępne. Pełne pochodzenie i digest modelu: [raport JSON](notes/results/sprint-3-native-benchmark.json).

## Przeznaczenie

Model wspiera demonstrację wyboru klientów kampanii telefonicznej: przewiduje,
czy klient założy lokatę terminową (`y=yes`, kod **1**), zanim znany jest wynik kontaktu.
Użytkownikiem demo jest osoba analizująca kampanię; wynik nie stanowi decyzji kredytowej
ani gotowej reguły biznesowej. W formularzu używamy danych fikcyjnych.
Nie oceniono skuteczności dla współczesnej kampanii ani innych populacji.

## Dane i wejście

Bank Marketing, UCI `bank-full.csv`, 45 211 rekordów; [kontrakt i źródło](datasets.md#bank_marketing--bank-marketing-bank-full),
DOI `10.24432/C5K306`, CC BY 4.0. Fingerprint wczytanych danych (`loaded-data-csv-v1`):
`2a81c082b2a35c24ae36ba337231bd88e4a417d0a5e02346169078fac5a92253`.

Split zachowuje kolejność chronologiczną: pierwsze 80% (36 168) do development,
ostatnie 20% (9043) do testu. Validation to trzy kolejne okna wewnątrz development,
każde po treningu wyłącznie na wcześniejszych rekordach. Bagging nie dotyczy tego modelu.
Udział `yes`: 2433/36 168 = 6,73% w development, 2856/9043 = 31,58% w teście.

Cechy liczbowe: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
Kategoryczne: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`,
`month`, `poutcome`. `duration` jest wykluczone, bo staje się znane po rozmowie.
Wspólna normalizacja traktuje `unknown` i `pdays=-1` jako braki. Normalizację wejścia
wykonuje się przed modelem; preprocessing wymagający `fit` jest zapisany w Pipeline.

## Ocena

| Metryka / liczność | Kandydat (zamrożony próg) | Dummy most-frequent, referencja holdout |
| --- | ---: | ---: |
| F1 | 0,5210 | 0,0000 |
| Precision | 0,3955 | 0,0000 |
| Recall | 0,7630 | 0,0000 |
| Accuracy | 0,5569 | 0,6842 |
| TN / FP / FN / TP | 2857 / 3330 / 677 / 2179 | 6187 / 0 / 2856 / 0 |

Oba wiersze modeli korzystają z tego samego testu 9043 rekordów; Dummy pochodzi
z runu holdout `20261123T232543-holdout-6bf251c7`. Średnie F1 okien baseline:
0,1707 (poszczególne: 0,1367 / 0,0977 / 0,2778). Progi i konfiguracje wybrano przed
końcowym testem, ale test był już oglądany w eksploracjach — to benchmark, nie nowy audyt.
Koszty wczytywania, predykcji, natywnego rozmiaru i RSS: [raport porównania](notes/sprint-3-model-comparison.md).
Pomiary paczki produkcyjnej pozostają do wykonania po #20.

## Ograniczenia

Duży dryf udziału klasy pozytywnej i rozrzut wyników kolejnych okien utrudniają
przewidywanie jakości w nowym okresie. Niepewność wyników nie została oszacowana
przedziałami ufności. Precision 0,3955 oznacza wiele zbędnych kontaktów przy obecnym
progu; F1 nie uwzględnia kosztu takiego kontaktu ani utraconej szansy. Nie oceniono
kalibracji prawdopodobieństw ani jakości w podgrupach klientów.

Dane są historyczne, zawierają braki i atrybuty finansowe. Nie mają identyfikatora
klienta, więc nie zweryfikowano powtarzających się osób pomiędzy okresami. Model
obsługuje technicznie nieznane kategorie, ale nie gwarantuje jakości dla nowych
kategorii ani rozkładów. Zmiana progu, cech lub preprocessingu wymaga nowej oceny
oraz zgodnych metadanych paczki.
