# Dataset Statistics

Generated directly from the dataset file by `src/run_experiments.py`.
Every number below is measured, not assumed.

- Source file: `<synthetic>`
- Rows in file: 900
- Rows used: 900
- Body column: `body`
- Label column: `label`
- Label encoding: numeric 0/1 as stored in file
- Label mapping: `{'1': 1, '0': 0}`

## Class distribution

| Label | Class | Count | Share |
|---|---|---|---|
| 0 | legitimate | 450 | 50.00% |
| 1 | phishing | 450 | 50.00% |

## Data quality and text size

| Quantity | Value |
|---|---|
| Missing email bodies | 0 |
| Rows dropped for missing label | 0 |
| Exact duplicate bodies | 40 |
| Mean body length (characters) | 146.4 |
| Median body length (characters) | 146.0 |
| Mean body length (whitespace tokens) | 21.5 |
| Mean length, legitimate (characters) | 143.9 |
| Mean length, phishing (characters) | 148.8 |
| TF-IDF vocabulary (fit on training split only) | 112 |
| TF-IDF feature dimension | 112 |
| TF-IDF n-gram range | (1, 1) |
| Distinct stop words removed | 318 |
| Distinct raw whitespace tokens | 155 |

Note: the TF-IDF vocabulary is fitted on the training split only, so the
reported dimension is the one the classifiers actually saw.
