# ⌛️ FIERLENIUS 🦙

✍ Authors:

**ROUAUD Lucas**

Master 2 Bio-informatics at *Univerité de Paris*

[![Python 3.10.8](https://img.shields.io/badge/python-%E2%89%A5_3.10.8-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/miniconda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/FilouPlains/FIERLENIUS.svg)](https://github.com/FilouPlains/FIERLENIUS)
![GitHub stars](https://img.shields.io/github/stars/FilouPlains/FIERLENIUS.svg?style=social)


## 🧮 Dataset origin

- **`SCOPe_2.08_classification.txt` :**
    - Extract from the `SCOPe 2.08` database, download the `01/2023`. Contains the classifications with the category. Only globular domains are kept.
- **`SCOPe_2.08_95identity_globular.fasta` :**
    - Extract from the `SCOPe 2.08` database, download the `02/2023`. Contains the classified sequences. Only globular domains are kept.
- **`pyHCA_SCOPe_30identity.out` :**
    - Extract from this Github repository: [DarkVador-HCA/Order-Disorder-continuum/blob/main/data/SCOPe/hca.out](https://github.com/DarkVador-HCA/Order-Disorder-continuum/blob/main/data/SCOPe/hca.out).
- **`HCDB_2018_summary_rss.csv` :**
    - Local database, version `02/2018`.
- **`peitsch2vec/` :**
    - Generate data with the script [`src/embeddings/peitsch2vec.py`](src/embeddings/peitsch2vec.py). In `peitsch2vec/default_domain/`, data have been generated without the option `--segment`, while in `peitsch2vec/default_segments/`, data have been generated with the option `--segment`.

## 🌲 Dependencies tree

```bash
$ tree -lF -h
.
├── [4.0K]  "data/"
│   ├── [ 15M]  "dir.des.scope.2.08-stable.txt"
│   ├── [4.4M]  "hca.out"
│   ├── [4.6K]  "HCDB_summary.csv"
│   └── [4.0K]  "peitsch2vec/"
│       ├── [4.0K]  "default_domain/"
│       │   ├── [ 26K]  "characteristics_2023-02-14_15-40-13.npy"
│       │   ├── [762K]  "embedding_2023-02-14_15-40-13.npy"
│       │   ├── [413K]  "matrix_cosine_2023-02-14_15-40-13.npy"
│       │   └── [776K]  "model_2023-02-14_15-40-13.w2v"
│       └── [4.0K]  "default_segments/"
│           ├── [ 26K]  "characteristics_2023-02-14_15-40-13.npy"
│           ├── [762K]  "embedding_2023-02-14_15-40-13.npy"
│           ├── [413K]  "matrix_cosine_2023-02-14_15-40-13.npy"
│           └── [776K]  "model_2023-02-14_15-40-13.w2v"
├── [4.0K]  "docs/"
│   ├── [4.0K]  "embedding/"
│   │   ├── [5.0M]  "comparing_distribution.html"
│   │   ├── [8.6M]  "matrix.html"
│   │   └── [5.0M]  "projection.html"
│   ├── [ 15K]  "index.html"
│   ├── [7.2K]  "jupyter_logo_icon.svg"
│   ├── [2.3K]  "style.css"
│   └── [4.0K]  "svg/"
│       ├── [ 27K]  "CBOW.svg"
│       ├── [2.2K]  "context_scheme.svg"
│       └── [5.1K]  "domain_to_peitsch.svg"
├── [4.0K]  "env/"
│   ├── [ 369]  "fierlenius.yml"
│   └── [ 740]  "README.md"
├── [1.5K]  "README.md"
└── [4.0K]  "src/"
    ├── [4.0K]  "embeddings/"
    │   ├── [6.6K]  "arg_parser.py"
    │   ├── [3.3K]  "domain.py"
    │   ├── [2.2K]  "hca_reader.py"
    │   ├── [1.1K]  "hcdb_parser.py"
    │   ├── [4.0K]  "notebook/"
    │   │   ├── [6.6M]  "comparing_distribution.ipynb"
    │   │   ├── [ 15M]  "matrix.ipynb"
    │   │   ├── [5.8M]  "projection.ipynb"
    │   │   └── [6.3K]  "sammon.py"
    │   ├── [7.4K]  "peitsch2vec.py"
    │   └── [4.8K]  "peitsch.py"
    └── [4.0K]  "msa/"
```
