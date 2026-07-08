# PHILM: Phage-Host Interaction Learning from Metagenomic profiles (macOS version)

## Overview

**PHILM** is a deep learning framework designed to predict phage-host interactions (PHIs) directly from metagenomic profiles and classify sample status using model-derived latent representations.

**Workflow**

![Workflow](img/Figure1.jpg)

## Table of Contents

- [PHILM: Phage-Host Interaction Learning from Metagenomic profiles (macOS version)](#philm-phage-host-interaction-learning-from-metagenomic-profiles-macos-version)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation on macOS](#installation-on-macos)
    - [Clone the Repository](#clone-the-repository)
      - [Option 1. Install using conda](#option-1-install-using-conda)
      - [Option 2. Install using pip inside a conda environment](#option-2-install-using-pip-inside-a-conda-environment)
  - [Usage](#usage)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Model Training](#2-model-training)
    - [3. Evaluation](#3-evaluation)
    - [4. Interaction Inference](#4-interaction-inference)
    - [5. Permutation-Based P-Value Estimation for Interactions Optional](#5-permutation-based-p-value-estimation-for-interactions-optional)
    - [6. Latent Representation Extraction Optional](#6-latent-representation-extraction-optional)
  - [Dependencies](#dependencies)
  - [License](#license)
  - [Citation](#citation)

## Installation on macOS

PHILM can be installed on macOS using either a conda environment file or a pip requirements file inside a conda environment.

### Clone the Repository

```bash
git clone git@github.com:YiyanYang0728/PHILM.git
cd PHILM
```
#### Option 1. Install using conda

```bash
conda env create -f env/environment.mac.yml
conda activate PHILM-mac
```

#### Option 2. Install using pip inside a conda environment

```bash
conda create -n PHILM-mac python=3.12
conda activate PHILM-mac
pip install -r env/requirements.mac.txt
```

## Usage
### 1. Data Preparation

Prepare three inputs for `scripts/split_data.py`: a prokaryotic relative abundance profile, a phage relative abundance profile, and an output directory.
Cross-kingdom relative abundance profiles, including prokaryotes and phages, can be generated using tools such as [sylph](https://github.com/bluenote-1577/sylph) and [phanta](https://github.com/bhattlab/phanta).

```bash
mkdir -p raw_data
curl -L --retry 3 -o raw_data/example.zip https://zenodo.org/records/21252847/files/example.zip
unzip -j raw_data/example.zip -d raw_data
rm raw_data/example.zip
```

Split data into training, validation and test datasets

```bash
python scripts/split_data.py --bact-arc raw_data/Bact_arc_profile.tsv --phage raw_data/Phage_profile.tsv --outdir data
```

**Outputs:**

* Raw split data without normalization:
  `data/<Phage/Bact_arc>_<train/validation/test>_no_clr.tsv`

* Normalized split data:
  `data/<Phage/Bact_arc>_<train/validation/test>.tsv`

* Feature names:
  `data/<Phage/Bact_arc>_feature_names.txt`

* Sample names:
  `data/<train/validation/test>_samples.txt`

### 2. Model Training

Configure training parameters and file paths in a YAML file.

```bash
# Copy the template configuration file and customize it
cp config/config_train.yaml my_config_train.yaml

# Run model training
python model/train.py -c my_config_train.yaml
```

Real-time training progress, checkpoints, and logs are saved in `checkpoints/`.

**Outputs:**

* Best model:
  `results/PHILM_best_model.pth`

* Best parameters:
  `results/PHILM_best_params.yaml`

* Checkpoints and logs:
  `checkpoints/`

### 3. Evaluation

Generate predictions on the test data.

```bash
# Prepare the test configuration file by merging the best parameters
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_test.yaml - \
    > my_config_test.yaml

# Run testing
python model/test.py -c my_config_test.yaml

# Summarize metrics
python scripts/summarize.py \
    results/PHILM_predict_test \
    &> results/PHILM_predict_test.ft.metrics
```

> **Tip:** Repeat the above steps for the validation and training sets by using `config_test.val.yaml` and `config_test.train.yaml`, respectively.

```bash
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_test.val.yaml - \
    > my_config_test.val.yaml

python model/test.py -c my_config_test.val.yaml

python scripts/summarize.py \
    results/PHILM_predict_val \
    &> results/PHILM_predict_val.ft.metrics
```

```bash
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_test.train.yaml - \
    > my_config_test.train.yaml

python model/test.py -c my_config_test.train.yaml

python scripts/summarize.py \
    results/PHILM_predict_train \
    &> results/PHILM_predict_train.ft.metrics
```

### 4. Interaction Inference

Infer host sensitivity scores for each phage feature.

```bash
# Infer interactions on the training dataset
# Note: my_config_test.train.yaml must already exist
python scripts/infer_interactions.py \
    -c my_config_test.train.yaml \
    --data-dir data \
    --split train \
    --score-mode gradient \
    --phage-normalize signed_zscore \
    --out results/PHILM_interactions.tsv
```

Optionally, interactions can be inferred using the combined training, validation, and testing datasets. This result is expected to be similar to the training-set-based result.

```bash
python scripts/infer_interactions.py \
    -c my_config_test.train.yaml \
    --data-dir data \
    --split all \
    --score-mode gradient \
    --phage-normalize signed_zscore \
    --out results/PHILM_interactions.all.tsv
```

**Output:**

* Predicted interaction scores:
  `results/PHILM_interactions.tsv`

### 5. Permutation-Based P-Value Estimation for Interactions Optional

This optional step estimates empirical p-values and adjusted p-values for PHILM-inferred interactions using permutation-based null models.

This step is computationally intensive and is recommended only when formal false discovery rate control is required. It is best suited for high-performance computing environments or systems with sufficient CPU resources. For exploratory analyses, users may first apply a heuristic cutoff, such as `normalized_score >= 3.5`, to prioritize candidate PHIs.

```bash
# Step 1: Re-train models using permuted data

cat config/config_train_perm.yaml \
    <(awk '{print "  "$0}' results/PHILM_best_params.yaml) \
    > train_perm.yaml

N=99
# Use N=999 for a larger permutation null when computational resources allow.
d=$PWD
seq 0 $N \
    | awk -v d=$d '{print "python "d"/model/train_permutation.py -c train_perm.yaml --perm-id "$1}' \
    > jobs.1.txt

# Run jobs.1.txt
while IFS= read -r cmd
do
  eval "$cmd"
done < jobs.1.txt
```

```bash
# Step 2: Infer PHIs from permutation-trained models

mkdir -p permutation_infer_configs

for i in $(seq -f "%03g" 0 $N); do
cat > permutation_infer_configs/config_infer_gradient.perm_${i}.yaml <<EOF
data:
  train_X: data/Phage_train.tsv
  train_Y: data/Bact_arc_train.tsv
  val_X: data/Phage_val.tsv
  val_Y: data/Bact_arc_val.tsv
  test_X: data/Phage_test.tsv
  test_Y: data/Bact_arc_test.tsv

model:
  path: results/permutation_null/perm_${i}/PHILM_perm_${i}.pth
  batch_size: 512
EOF

awk '$0~/hidden_size/{print "  "$0}' results/PHILM_best_params.yaml \
    >> permutation_infer_configs/config_infer_gradient.perm_${i}.yaml
done
```

```bash
echo -n "" > jobs.2.txt

for f in permutation_infer_configs/config_infer_gradient.perm_*.yaml; do
    i=$(basename "$f" .yaml | cut -f2 -d. | cut -f2 -d_)

    echo "python $d/scripts/infer_interactions.py \
        -c permutation_infer_configs/config_infer_gradient.perm_${i}.yaml \
        --data-dir data \
        --split train \
        --score-mode gradient \
        --sort-by none \
        --phage-normalize signed_zscore \
        --out results/permutation_null/perm_${i}/PHILM_perm_${i}.raw_gradient.tsv" \
        >> jobs.2.txt
done

# Run jobs.2.txt
while IFS= read -r cmd
do
  eval "$cmd"
done < jobs.2.txt
```

```bash
# Step 3: Calculate empirical p-values and adjusted p-values

python scripts/permutation_pvalues.py \
    --norm_mode off \
    --observed results/PHILM_interactions.tsv \
    --perm_glob "results/permutation_null/perm_*/PHILM_perm_*.raw_gradient.tsv" \
    --out results/PHILM_interactions.pvalues.tsv
```

Filter significant interactions by adjusted p-value.

```bash
awk -F"\t" 'NR==1 || $5<0.05' \
    results/PHILM_interactions.pvalues.tsv \
    | cut -f1,2,3,5 \
    > results/PHILM_interactions.pvalues.filtered.tsv
```

### 6. Latent Representation Extraction Optional

PHILM can extract sample-level latent representations from a trained model. These representations can be used for downstream analyses such as sample classification, clustering, and visualization.

```bash
# Determine the output dimension
outdim=$(wc -l < data/Bact_arc_feature_names.txt)

# Prepare the representation extraction configuration file
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_repr.train.yaml - \
    > my_config_repr.train.yaml

sed -i "s|#OUTDIM#|$outdim|g" my_config_repr.train.yaml

# Extract representations for the training data
python model/extract_repr.py -c my_config_repr.train.yaml
```

> **Tip:** Apply the same process for the validation and test sets by using `config_repr.val.yaml` and `config_repr.test.yaml`.

```bash
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_repr.val.yaml - \
    > my_config_repr.val.yaml

sed -i "s|#OUTDIM#|$outdim|g" my_config_repr.val.yaml

python model/extract_repr.py -c my_config_repr.val.yaml
```

```bash
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat config/config_repr.test.yaml - \
    > my_config_repr.test.yaml

sed -i "s|#OUTDIM#|$outdim|g" my_config_repr.test.yaml

python model/extract_repr.py -c my_config_repr.test.yaml
```

Merge representations.

```bash
cat results/PHILM_train_repr1.tsv \
    results/PHILM_val_repr1.tsv \
    results/PHILM_test_repr1.tsv \
    > results/repr1_all.tsv

cat results/PHILM_train_repr2.tsv \
    results/PHILM_val_repr2.tsv \
    results/PHILM_test_repr2.tsv \
    > results/repr2_all.tsv
```

The `repr<1/2>_all.tsv` files are tab-separated files without row names or column names. The number of columns corresponds to the dimensionality of the PHILM-derived representations. Samples are organized as rows and are ordered consistently with `data/train_samples.txt`, `data/val_samples.txt`, and `data/test_samples.txt`.

## Dependencies

PHILM requires the following major Python packages:

* PyTorch
* torchdiffeq
* NumPy
* pandas
* SciPy
* scikit-learn
* scikit-bio
* PyYAML
* Optuna

Please see the requirement files in `env/` for the full dependency lists.

## License

This project is released under the [MIT License](./LICENSE).

## Citation

Please cite PHILM as follows:

Yang, Y., Wang, T., Huang, D., Wang, X. W., Weiss, S. T., Korzenik, J., & Liu, Y. Y. (2025). *Deep Learning Transforms Phage-Host Interaction Discovery from Metagenomic Data*. bioRxiv.
