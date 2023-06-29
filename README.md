# ⌛️ FIERLENIUZ 🦙

[![Python 3.10.8](https://img.shields.io/badge/python-%E2%89%A5_3.10.8-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/miniconda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/FilouPlains/FIERLENIUZ.svg)](https://github.com/FilouPlains/FIERLENIUZ)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
![GitHub stars](https://img.shields.io/github/stars/FilouPlains/FIERLENIUZ.svg?style=social)

✍ Authors: **ROUAUD Lucas**

📚 Formation: Master 2 Bio-informatics at *Univerité de Paris*

## ⚙️ Installation

### 🤓 Lazy methods

A script called `INSTALL.sh` is made to facilitate this script installation. When you are in the root project directory `📁 ./`, use:

```bash
bash INSTALL.sh
```

All used commands are described in the next parts (_**🧍‍♂️🧍‍♂️ Cloning the repository**_; _**📶 Install conda environment**_; _**📁⏩📂 Data decompression**_)!

### 🧍‍♂️🧍‍♂️ Cloning the repository

To clone the repository in your computer, use the next command:

```bash
git clone git@github.com:FilouPlains/FIERLENIUZ.git
cd FIERLENIUZ/
```

### 📶 Install conda environment

> This repository is using Python. To install packages, conda is used and you can refer to their website to install it: https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html

Once conda is installed (if it was not already the case), simply used those next commands to use the program (when you are in the root project directory `📁 ./`):

```bash
conda env create -n fierlenius -f env/fierlenius.yml
conda activate fierlenius
```

### 📁⏩📂 Data decompression

Some data were too heavy to be simply put like that into the repository. So they were compressed. So next commands have to be used (when you are in the root project directory `📁 ./`):

```bash
tar -xf data/peitsch2vec/default_domain.tar.gz -C data/peitsch2vec/
tar -xf data/peitsch2vec/redundancy/30_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/70_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/90_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
```

## 🌲 Dependencies tree

```bash
$ tree -lF -h

[4.0K]
.
├── [4.0K]  "data/"
│   ├── [4.0K]  "output_plot/"
│   ├── [4.0K]  "peitsch2vec/"
│   │   ├── [ 40M]  "default_domain.tar.gz"
│   │   └── [4.0K]  "redundancy/"
│   │       ├── [ 39M]  "30_percent_redundancy.tar.gz"
│   │       ├── [159M]  "70_percent_redundancy.tar.gz"
│   │       └── [248M]  "90_percent_redundancy.tar.gz"
│   └── [4.0K]  "REDUNDANCY_DATASET/"
├── [4.0K]  "env/"
├── [ 20K]  "LICENSE"
├── [3.8K]  "README.md"
└── [4.0K]  "src/"
    ├── [4.0K]  "embeddings/"
    │   ├── [4.0K]  "genetic_deep_learning/"
    │   └── [4.0K]  "notebook/"
    ├── [4.0K]  "hca_extraction/"
    └── [4.0K]  "scope_tree/"

17 directories, 84 files
```

_This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)._

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
