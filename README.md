# ⌛️ FIERLENIUZ 🦙

[![Python 3.10.8](https://img.shields.io/badge/python-%E2%89%A5_3.10.8-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/miniconda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/FilouPlains/FIERLENIUZ.svg)](https://github.com/FilouPlains/FIERLENIUZ)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
![GitHub stars](https://img.shields.io/github/stars/FilouPlains/FIERLENIUZ.svg?style=social)

✍ Authors: **ROUAUD Lucas**

📚 Formation: Master 2 Bio-informatics at *Univerité de Paris*

📏 This program were write during an internship in **Institut de Minéralogie, de Physique et de Cosmochimie, Sorbonne Université, UMR7590, CNRS, Muséum national d’Histoire naturelle**; in the **bioinformatique et biophysique** team.

_**📋 This work was supported by the French National Research Agency (ANR-21-CE12-0021).**_

## ⚙️ Installation

### 🤓 Lazy method

A script called `INSTALL.sh` is made to facilitate this script installation. To used it, do:

```bash
bash INSTALL.sh
```

All used commands are described in the next parts (_**Cloning the repository**_; _**Install conda environment**_; _**Data decompression**_)! This script is available at the release page: https://github.com/FilouPlains/FIERLENIUZ/releases/tag/v1.2.3

⚠️ After the installation, **DO NOT FORGET TO ACTIVATE YOUR CONDA ENVIRONMENT**:

```bash
conda activate fierleniuz
```

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
conda env create -n fierleniuz -f env/fierleniuz.yml
conda activate fierleniuz
```

### 📂 Data decompression

Some data were too heavy to be simply put like that into the repository. So they were compressed. So next commands have to be used (when you are in the root project directory `📁 ./`):

```bash
tar -xf data/peitsch2vec/default_domain.tar.gz -C data/peitsch2vec/
tar -xf data/peitsch2vec/redundancy/30_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/70_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/90_percent_redundancy.tar.gz -C data/peitsch2vec/redundancy/
```

## 💻 Implemented script

### 💾 The main script

To have a description of the parameters and an example of command, use this next one:

```bash
python src/embeddings/peitsch2vec.py -h
```

This script is used in order to transform a corpus of hydrophobic clusters into vectors.

### 💾 Secondary scripts

1. A script used to transform a `.fasta` file into a `.out` file. To have a description of the parameters and an example of command, use this next one:

```bash
python3 src/hca_extraction/hca_extraction.py -h
```

2. A script used do computed the context diversity ad output a `.csv` format. To have a description of the parameters and an example of command, use this next one:

```bash
python3 src/scope_tree/context_extraction.py -h
```

3. A script used to computed a network linked to SCOPe with context diversity coloration and context diversity distribution through plotly. To have a description of the parameters and an example of command, use this next one:

```bash
python3 src/scope_tree/scope_tree.py -h
```

### 🌐 External software

`cd-hit` is a software used to treated sequences redundancy. It is available at this next webpage (GitHub) https://github.com/weizhongli/cdhit/releases. To use it, type the following command is used:

```bash
cd-hit -i {scope_database}.fa -o cd-hit_{i}.fasta -c 1
```

With :
- **`{scope_database}.fa` :** Use here the original SCOPe database, with the sequence at `.fa` format. You can download the dataset here: https://scop.berkeley.edu/astral/subsets/ver=2.08. In `Percentage identity-filtered Astral SCOPe genetic domain sequence subsets, based on PDB SEQRES records`, use `sequences` and `less than 30`, `less than 70` and `less than 90` parameters.
- **`cd-hit_{i}.fasta` :** How to name the output file. For this respository, the output is named `cd-hit_30.fasta`, `cd-hit_70.fasta` and `cd-hit_90.fasta`.

### 🖥 Cluster accession

For some scripts, the cluster **PCIA - Plateforme de calcul intensif du Muséum national d’Histoire naturelle** have been used. The next script were used to launch the job (in this next cluster path `📁 STAGE_M2/`):

```bash
sbatch LAUNCH_SCRIPT.sh
```

**⚠️ TWO THINGS TO KNOW:**
1. Output are done in next directory: `📁 /mnt/beegfs/abruley/CONTEXT/`.
2. The script **HAVE TO BE MANUALLY EDITED** if you want to change the input database.

The used script is available at `📁 src/cluster/launch_script_90.sh`.

### 📜 Short description of the other script

1. **`📁 src/cluster/launch_script_90.sh` :** Script used on the cluster to computed the context diversity.
2. **`📁 src/embeddings/arg_parser.py` :** Parse given arguments for the `💾 The main script`.
3. **`📁 src/embeddings/context_analyzer.py` :** Compute ordered and unordered diversity contexts. There is also a function to extract and center words for a given window.
4. **`📁 src/embeddings/domain.py` :** UNUSED, deprecated.
5. **`📁 src/embeddings/genetic_deep_learning/correlation_matrix.py` :** Computed the correlation between two matrices.
6. **`📁 src/embeddings/genetic_deep_learning/genetic_algorithm.py` :** Genetic algorithms to select the best Word2Vec model.
7. **`📁 src/embeddings/genetic_deep_learning/hca_out_format_reader.py` :** Transform a whole `.out` into a corpus usable by Word2Vec.
8. **`📁 src/embeddings/genetic_deep_learning/running_model.py` :** Run a Word2Vec model.
9. **`📁 src/embeddings/hca_reader.py` :** Parse a `.out` file to extract information from it.
10. **`📁 src/embeddings/hcdb_parser.py` :** Parse the hydrophobic cluster database.
11. **`📁 src/embeddings/notebook/comparing_distribution.ipynb` :** Plot of the distribution of some characteristics using plotly.
12. **`📁 src/embeddings/notebook/data_meaning.ipynb` :** Plot information like mostly to the norm using Plotly.
13. **`📁 src/embeddings/notebook/matplotlib_for_report.ipynb` :** Used matplotlib to produced `plot.pdf` to use into the report.
14. **`📁 src/embeddings/notebook/matrix.ipynb` :** Computed cosine similarities matrix.
15. **`📁 src/embeddings/notebook/projection.ipynb` :** Test a lot of projection for the vectors, with a lot of descriptors.
16. **`📁 src/embeddings/notebook/sammon.py` :** Computed a sammon map using this next GitHub repository: https://github.com/tompollard/sammon.
17. **`📁 src/embeddings/peitsch2vec.py` :** The main program used to computed Word2Vec vectors and other characteristics.
18. **`📁 src/embeddings/peitsch.py` :** Object to manipulate the hydrophobic clusters.
19. **`📁 src/embeddings/write_csv.py` :** Write a `.csv` file with some hydrophobic clusters characteristics.
20. **`📁 src/hca_extraction/arg_parser.py` :** Parse given arguments for the `📁 src/hca_extraction/hca_extraction.py` script.
21. **`📁src/hca_extraction/domain_comparison.py` :** Script used to compared multiple domain between them by computing the context diversity and output the best result through an user define threshold.
22. **`📁 src/hca_extraction/hca_extraction.py` :** Go from a `.fasta` files to a `.out` file.
23. **`📁 src/scope_tree/arg_parser.py` :** Parse given arguments for the `📁 src/scope_tree/context_extraction.py` and `📁 src/scope_tree/scope_score.py` scripts.
24. **`📁 src/scope_tree/context_extraction.py` :** Extract the context informations, taking also in consideration the SCOPe levels, and output a `.csv` file.
25. **`📁 src/scope_tree/scope_score.py` :** Computed a score between two or multiple domains to see how far they are from each other in the SCOPe tree.
26. **`📁 src/scope_tree/scope_tree.py` :** Computed a network of one given hydrophobic clusters. The network is linked to the SCOPe tree, with the indications of the context diversity on each nodes.

## 📊 Dataset quick description

1. **`📁 data/HCDB_2018_summary_rss.csv` :** Hydrophobic clusters database with the summary of the regular secondary structures. Made in 2018.
2. **`📁 pyHCA_SCOPe_30identity_globular.out` :** pyHCA output. It were applied on the SCOPe `2.07` database with a redundancy level of 30 %, download trough Astral. Original dataset available here: https://raw.githubusercontent.com/DarkVador-HCA/Order-Disorder-continuum/main/data/SCOPe/hca.out.
3. **`📁 SCOPe_2.08_classification.txt` :** A file that permits to go from the domain ID to the SCOPe precise class (for instance, from `d1ux8a_` to `a.1.1.1`). File is available here: https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.08-stable.txt.
4. **`📁 output_plot/` :** All plots produced by the notebook `src/embeddings/notebook/matplotlib_for_report.ipynb`, all in `.pdf` format.
5. **`📁 data/REDUNDANCY_DATASET/cd-hit_30.fasta`; `📁 data/REDUNDANCY_DATASET/cd-hit_70.fasta`; `📁 data/REDUNDANCY_DATASET/cd-hit_90.fasta` :** Amino acids sequences from SCOPe `2.08` with different redundancy levels (30 %, 70 %, 90 %). Redundancy were treated through Astral and cd-hit. Original dataset are available here: https://scop.berkeley.edu/astral/subsets/ver=2.08.
6. **`📁 data/REDUNDANCY_DATASET/cd-hit_30.out`; `📁 data/REDUNDANCY_DATASET/cd-hit_70.out`; `📁 data/REDUNDANCY_DATASET/cd-hit_90.out` :** Hydrophobic clusters sequences from SCOPe `2.08` with different redundancy levels (30 %, 70 %, 90 %). Redundancy were treated through Astral and cd-hit. Not treated by pyHCA.
7. **`📁 data/REDUNDANCY_DATASET/redundancy_30_context_conservation_2023-05-09_14-38-42.csv`; `📁 data/REDUNDANCY_DATASET/redundancy_70_context_conservation_2023-05-11_10-39-29.csv`; `📁 data/REDUNDANCY_DATASET/redundancy_90_context_conservation_2023-05-11_10-41-19.csv` :** All context diversity computed for different redundancy levels (30 %, 70 %, 90 %). Redundancy were treated through Astral and cd-hit. _**Little things to know:** `100.0 =` context computed with a full diversity; `100 =` context could not be computed, so a full diversity have been attributed. This have been corrected in the program by putting `NA` instead of "`int(100)`"._
8. **`📁 data/peitsch2vec/default_domain/` :** Data for the dataset with a redundancy level of 30 %, treated by pyHCA, not treated by cd-hit.
9. **`📁 data/peitsch2vec/redundancy/30_percent_redundancy/`; `📁 data/peitsch2vec/redundancy/70_percent_redundancy/`; `📁 data/peitsch2vec/redundancy/90_percent_redundancy/` :** Data for the dataset with a redundancy level of 30 %, 70 %, 90 %, not treated by pyHCA, treated by cd-hit.

**For the path given in `8.` and `9.`:**

10. **`📁 characteristics_`{date}`.npy/` :** Hydrophobic clusters characteristics for a given redundancy level, like the size or the regular secondary structure. The characteristics are listed here, in the same order as this file:
    1. Peitch code.
    2. Hydrophobic cluster (binary code).
    3. Hydrophobic score.
    4. Cluster size.
    5. Regular secondary structure.
    6. Occurences.
    7. Number of cluster inside the domain, were the cluster[i] is found.
    8. Domain size, were the cluster[i] is found.
    9. Score HCA define by pyHCA, were the cluster[i] is found.
    10. P-value define by pyHCA, were the cluster[i] is found.
11. **`📁 corpus_`{date}`.npy/` :** Corpus given to Word2Vec, after applying the filters.
12. **`📁 embedding_`{date}`.npy/` :** Vector embedding generated by Word2Vec.
13. **`📁 matrix_cosine_`{date}`.npy/` :** Cosine similarities matrix generated from the vector embedding, generated by Word2Vec.
14. **`📁 model_`{date}`.w2v/` :** The trained Word2Vec models.

## 🌲 Dependencies tree

```bash
$ tree -lF -h

[6.8G]
.
├── [4.0K]  "data/"
│   ├── [4.0K]  "output_plot/"
│   ├── [4.0K]  "peitsch2vec/"
│   │   ├── [4.0K]  "default_domain/"
│   │   └── [4.0K]  "redundancy/"
│   │       ├── [4.0K]  "30_percent_redundancy/"
│   │       ├── [4.0K]  "70_percent_redundancy/"
│   │       ├── [4.0K]  "90_percent_redundancy/"
│   └── [4.0K]  "REDUNDANCY_DATASET/"
├── [4.0K]  "env/"
│   ├── [ 905]  "fierleniuz.yml"
│   └── [ 885]  "README.md"
├── [ 895]  "INSTALL.sh"
├── [ 20K]  "LICENSE"
├── [ 13K]  "README.md"
└── [4.0K]  "src/"
  ├── [4.0K]  "cluster/"
  ├── [4.0K]  "embeddings/"
  │   ├── [4.0K]  "genetic_deep_learning/"
  │   └── [4.0K]  "notebook/"
  ├── [4.0K]  "hca_extraction/"
  └── [4.0K]  "scope_tree/"

18 directories, 88 files
```

_This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)._

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
