# Starting by cloning the repository.
echo "[[CLONING THE REPOSITORY]]"
git clone git@github.com:FilouPlains/FIERLENIUZ.git
cd FIERLENIUZ/

# Creating the conda environment.
echo "[[CREATING CONDA ENVIRONMENT]]"
conda env create -n fierleniuz -f env/fierleniuz.yml
conda activate fierleniuz
conda --version

# Decompressing archives.
echo "[[DECOMPRESSING ARCHIVES]]"
tar -xf data/peitsch2vec/default_domain.tar.gz -C data/peitsch2vec/
tar -xf data/peitsch2vec/redundancy/30_percent_redundancy.tar.gz \
    -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/70_percent_redundancy.tar.gz \
    -C data/peitsch2vec/redundancy/
tar -xf data/peitsch2vec/redundancy/90_percent_redundancy.tar.gz \
    -C data/peitsch2vec/redundancy/

# Testing if everything is good
python src/embeddings/peitsch2vec.py -h
