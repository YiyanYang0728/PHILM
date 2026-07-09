<div align="center" style="background-color:#316297; color:#ffffff; padding:42px 28px; border-radius:8px; margin-bottom:28px;">
  <h1 style="color:#ffffff; margin:0 0 18px 0;">PHILM Step-by-Step Tutorial</h1>
  <p style="color:#ffffff; font-size:18px; line-height:1.6; margin:0 auto 26px auto; max-width:980px;">
    Reproduce the analysis on 7,016 healthy human stool samples, from data preparation to interaction inference and permutation-based significance testing.
  </p>
  <p style="color:#ffffff; font-size:18px; margin:0;">
    <a href="#contents" style="color:#ffffff;">Contents</a> |
    <a href="#usage" style="color:#ffffff;">Usage</a> |
    <a href="#tutorial" style="color:#ffffff;">Tutorial</a> |
    <a href="#results" style="color:#ffffff;">Results</a>
  </p>
</div>

## Overview

**PHILM** is a deep learning framework designed to predict phage-host interactions (PHIs) directly from metagenomic profiles.
In this tutorial, we will teach you how to use PHILM step by step and reproduce the analysis of 7,016 healthy human stool samples reported in [our paper](https://www.biorxiv.org/content/10.1101/2025.05.26.656232v2.full).

Please follow the PHILM installation instructions on [GitHub](https://github.com/YiyanYang0728/PHILM). **Note**: Select the version that matches your Linux GPU, Linux CPU, or macOS system. In this tutorial, we use the Linux GPU-supported PHILM version.

## Usage

After installation, PHILM provides several command-line tools for data preparation, model training, evaluation, interaction inference, permutation-based significance testing, and more.

Available commands are listed below:

| Command             | Purpose                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `philm-split`       | Split paired prokaryotic and phage abundance profiles into training, validation, and test datasets. |
| `philm-train`       | Train a PHILM model using a YAML configuration file.                                                |
| `philm-evaluate`    | Evaluate a trained PHILM model and generate prediction results.                                     |
| `philm-summarize`   | Summarize prediction performance metrics from PHILM output files.                                   |
| `philm-interaction` | Infer phage-prokaryote interaction scores from a trained PHILM model.                               |
| `philm-permutate`   | (Optional) Train PHILM models on permuted datasets for null-model construction.                     |
| `philm-pval`        | (Optional) Calculate empirical p-values and adjusted p-values from permutation-based null results.  |
| `philm-repr`        | (Optional) Extract PHILM-derived latent representations from a trained model.                       |

To predict phage-host interactions, we will use the following commands in order: `philm-split`, `philm-train`, `philm-evaluate`, `philm-summarize`, `philm-interaction`, `philm-permutate`, and `philm-pval`.

## Tutorial

**Workflow:**

| Step | Section |
| ---- | ------- |
| Step 1 | [Prepare input files](#step-1-prepare-input-files) |
| Step 2 | [Train PHILM with phage profiles to predict prokaryotic profiles](#step-2-train-philm-with-phage-profiles-to-predict-prokaryotic-profiles) |
| Step 3 | [Evaluate the trained model](#step-3-evaluate-the-trained-model) |
| Step 4 | [Predict phage-host interactions using gradient-based sensitivity analysis](#step-4-predict-phage-host-interactions-using-gradient-based-sensitivity-analysis) |
| Step 5 | [Estimate permutation-based P-values for PHIs (Optional)](#step-5-estimate-permutation-based-p-values-for-phis-optional) |


### Step 1. Prepare input files

PHILM can directly take taxonomic relative abundance profiles as inputs. Usually, these inputs are (1) a prokaryotic relative abundance profile and (2) a phage relative abundance profile. These profiles should use samples as columns and taxa as rows, as shown below.

Prokaryotic profile:

| clade_name                           | ERR1018203 | ERR1293505 | ERR1293506 | ERR1293508 |
| ------------------------------------ | ---------- | ---------- | ---------- | ---------- |
| s__Methanocatella smithii            | 0.0        | 0.0        | 0.0        | 0.0        |
| s__Methanoprimaticola hominis        | 0.0        | 0.0        | 0.0        | 0.0        |
| s__Bifidobacterium adolescentis      | 0.0        | 0.0        | 0.0        | 0.0        |
| s__Bifidobacterium animalis          | 0.0        | 0.0        | 0.0        | 0.0        |
| s__Bifidobacterium bifidum           | 0.0        | 0.0        | 0.0        | 0.0        |
| ...                                  | ...        | ...        | ...        | ...        |

Phage profile:

| clade_name  | ERR1018203 | ERR1293505 | ERR1293506 | ERR1293508 |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| vOTU-014862 | 0.0        | 0.00095516 | 0.00058027 | 0.00030762 |
| vOTU-014915 | 0.0        | 0.00083192 | 0.00081714 | 0.00055150 |
| vOTU-014945 | 0.0        | 0.0        | 0.0        | 0.0        |
| vOTU-014971 | 0.0        | 0.00237088 | 0.00220980 | 0.00230800 |
| vOTU-014982 | 0.0        | 0.0        | 0.0        | 0.0        |
| ...         | ...        | ...        | ...        | ...        |


Prokaryotic and phage relative abundance profiles can be generated using tools such as [sylph](https://github.com/bluenote-1577/sylph) and [phanta](https://github.com/bhattlab/phanta).
Use sylph to generate a combined profile containing both prokaryotes and phages by running:

```bash
sylph profile gtdb-r226-c200-dbv1.syldb uhgv_c100_dbv1.syldb -c100 -1 <fastq_folder>/*_1.fastq.gz -2 <fastq_folder>/*_2.fastq.gz -o <output_folder>/prok_phage_profile.tsv
```

Alternatively, generate the prokaryotic and phage profiles separately by using a different reference each time:

```bash
sylph profile gtdb-r226-c200-dbv1.syldb -c100 -1 <fastq_folder>/*_1.fastq.gz -2 <fastq_folder>/*_2.fastq.gz -o <output_folder>/prok_profile.tsv
sylph profile uhgv_c100_dbv1.syldb -c100 -1 <fastq_folder>/*_1.fastq.gz -2 <fastq_folder>/*_2.fastq.gz -o <output_folder>/phage_profile.tsv
```

Here, we use the references `gtdb-r226-c200-dbv1.syldb` and `uhgv_c100_dbv1.syldb` for prokaryotes and phages, respectively. These references are available from the [sylph pre-built DBs](https://sylph-docs.github.io/pre%E2%80%90built-databases/).

Phanta only generates cross-kingdom abundance profiles, so you have to split them to obtain separate prokaryotic and phage profiles.

For your convenience, we have provided the sylph-derived species-level taxonomic relative abundance profiles (694 prokaryotic species + 2,102 viral species) on Zenodo.
**Note**: We filtered the taxa to retain those with prevalence > 5%, so that the number of samples is greater than or equal to the number of taxa, which is a prerequisite for training a reliable PHILM model.
Download and decompress the data:

```bash
mkdir -p raw_data
wget -t 3 -O raw_data/tutorial_data.zip https://zenodo.org/records/21269560/files/tutorial_data.zip
unzip -j raw_data/tutorial_data.zip -d raw_data
rm raw_data/tutorial_data.zip
```

Next, use `philm-split` to split the paired profiles into training, validation, and test datasets and perform normalization:

```bash
philm-split --bact-arc raw_data/Bact_arc_profile.tsv --phage raw_data/Phage_profile.tsv --outdir data
```

The default training:validation:test sample ratio is 8:1:1. You can modify this ratio if necessary. Use `philm-split -h` for more help.

**Outputs:**

The outputs are placed in the `data/` folder, which contains multiple files. The most important files are `data/<Phage/Bact_arc>_<train/validation/test>.tsv`. These files are tab-delimited and do not include row names or column names. Their dimensions are number of samples (rows) x number of taxa (columns). The taxa names are stored in `data/<Phage/Bact_arc>_feature_names.txt` in the same order as the columns, while the sample names are stored in `data/<train/validation/test>_samples.txt` in the same order as the rows.

* Raw split data without normalization:
  `data/<Phage/Bact_arc>_<train/validation/test>_no_clr.tsv`

* Normalized split data:
  `data/<Phage/Bact_arc>_<train/validation/test>.tsv`

* Feature names:
  `data/<Phage/Bact_arc>_feature_names.txt`

* Sample names:
  `data/<train/validation/test>_samples.txt`

### Step 2. Train PHILM with phage profiles to predict prokaryotic profiles

In general, PHILM uses paired phage profiles as input and predicts the corresponding prokaryotic profiles as output.
`philm-train` accepts a YAML configuration file where you can define input paths and training parameters.
A typical YAML configuration file has two parts:

```yaml
data:
  train_X: data/Phage_train.tsv
  train_Y: data/Bact_arc_train.tsv
  val_X: data/Phage_val.tsv
  val_Y: data/Bact_arc_val.tsv
model:
  batch_size: 512 # Batch size
  patience: 20 # Number of epochs with no improvement after which training will be stopped
  num_epochs: 8000 # Maximum number of epochs for training
  n_trials: 10 # Number of trials for choosing the optimal combination of parameters, default=10
  path: results/PHILM_best_model.pth # Where the best model should be saved
  para_path: results/PHILM_best_params.yaml # Where the best model's parameters should be saved
```

`data` records the input paths. Both absolute and relative paths work. `model` sets the training parameters and output paths.

We prepared a template named `src/philm/config/config_train.yaml`, and you can copy it to your working directory. If you use a different output folder for `philm-split`, remember to change the paths in the YAML file. Here, we copy it in the `PHILM/` directory:

```bash
# Copy the template configuration file and customize it
cp src/philm/config/config_train.yaml my_config_train.yaml
```

Then, feed this YAML file to `philm-train` to start training:

```bash
# Run model training
philm-train -c my_config_train.yaml
```

**Note**: Training time varies across systems. On a Linux GPU system with data containing > 7,000 samples and > 2,700 species, PHILM typically takes a few hours to finish. It is better to run this job in the background with `nohup philm-train -c my_config_train.yaml &` or submit it as an HPC job.

According to the configuration file, the deep learning model reads 512 samples each time (`batch_size=512`) and either runs for 8,000 epochs if the validation loss keeps dropping (`num_epochs=8000`) or stops early if the validation loss does not improve after 20 epochs (`patience=20`). The model identifies the optimal model by trying 10 different sets of parameters (`n_trials=10`).

For real-time training progress, go to `checkpoints/`, where checkpoint models and logs are saved. Specifically, check `checkpoints/train_0.log` to monitor the training and validation loss in each epoch:

```text
2026-06-08 00:08:52 - INFO - MLP_NODE(
  (fc1): Linear(in_features=699, out_features=512, bias=True)
  (ode_func): ODEFunc(
    (model): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=True)
      (1): Tanh()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): Tanh()
    )
  )
  (fc2): Linear(in_features=512, out_features=100, bias=True)
)
2026-06-08 00:08:56 - INFO - Epoch 1/8000, Loss: 1.120550, Val_loss: 1.330073
2026-06-08 00:08:56 - INFO - New best model saved with val loss: 1.330073
2026-06-08 00:08:57 - INFO - Epoch 2/8000, Loss: 1.117192, Val_loss: 1.327149
2026-06-08 00:08:57 - INFO - New best model saved with val loss: 1.327149
2026-06-08 00:08:57 - INFO - Epoch 3/8000, Loss: 1.113972, Val_loss: 1.324241
2026-06-08 00:08:57 - INFO - New best model saved with val loss: 1.324241
... (trying 10 models)
2026-06-08 01:58:45 - INFO - Epoch 1210/8000, Loss: 0.133390, Val_loss: 0.631046
2026-06-08 01:58:45 - INFO - Early stopping triggered after 1210 epochs. The best validation_loss is 0.6309853792190552
```

The log first shows the model structure for each trial, followed by the detailed training loss (`Loss`) and validation loss (`Val_loss`) for each epoch. The log also notifies you when the current model reaches a better validation loss and explains why training stops.

**Outputs:**

* Best model:
  `results/PHILM_best_model.pth`

* Best model's parameters:
  `results/PHILM_best_params.yaml`

* Checkpoints and logs:
  `checkpoints/model_trial_<trial_number>.pth` and `checkpoints/train_0.log`

### Step 3. Evaluate the trained model

Next, evaluate the trained model on the test data. Likewise, we provide a template YAML file, `src/philm/config/config_test.yaml`. Add the best model's path and parameters to this file:

```bash
# Prepare the evaluation configuration file by merging the best parameters
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat src/philm/config/config_test.yaml - \
    > my_config_test.yaml
```

`my_config_test.yaml` should look like this:

```yaml
data:
  test_X: data/Phage_test.tsv
  test_Y: data/Bact_arc_test.tsv
output_prefix: results/PHILM_predict_test
model:
  path: results/PHILM_best_model.pth
  batch_size: 512
  hidden_size: <Best model's parameter>
  learning_rate: <Best model's parameter>
  weight_decay: <Best model's parameter>
```

`philm-evaluate` runs the trained model on the test data and generates evaluation outputs using the configured output prefix.
`philm-summarize` summarizes the evaluation metrics into `results/PHILM_predict_test.ft.metrics`.

```bash
# Run evaluation on the test dataset
philm-evaluate -c my_config_test.yaml
# Summarize metrics
philm-summarize results/PHILM_predict_test &> results/PHILM_predict_test.ft.metrics
```

A glimpse of `results/PHILM_predict_test.ft.metrics`:

```text
#############feature-wise metrics#############
Mean PCC: 0.65565781685717
No. of pcc > 0.8: 122
Perc of pcc > 0.8: 17.57925072046109510000
Top 50 pcc mean: 0.88270095782787
Top 20 pcc mean: 0.90323705603244
Top 10 pcc mean: 0.91300208489238
...
```

The most important metric is `Mean PCC` in `feature-wise metrics`. It calculates the average Pearson correlation coefficient (PCC) of prokaryotic taxa across samples between the predicted and ground truth data.

> **Tip:** Similarly, you can repeat the steps above for the validation and training sets by creating `config_test.val.yaml` and `config_test.train.yaml`, respectively.

```bash
# Run evaluation on the validation dataset
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat src/philm/config/config_test.val.yaml - \
    > my_config_test.val.yaml
philm-evaluate -c my_config_test.val.yaml
philm-summarize results/PHILM_predict_val &> results/PHILM_predict_val.ft.metrics
# Run evaluation on the training dataset
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat src/philm/config/config_test.train.yaml - \
    > my_config_test.train.yaml
philm-evaluate -c my_config_test.train.yaml
philm-summarize results/PHILM_predict_train &> results/PHILM_predict_train.ft.metrics
```

### Step 4. Predict phage-host interactions using gradient-based sensitivity analysis

PHILM infers phage-host interactions (PHIs) by measuring how sensitive each predicted prokaryotic abundance is to each phage input. After training, automatic differentiation is used to calculate these sensitivities.
In short, the gradient-based sensitivity analysis asks: "If I slightly change the abundance of one phage, how much does the model's predicted abundance of one bacterium change?"
These sensitivities are averaged across samples and normalized within each phage. A high positive score suggests a potential phage-host association.

**Note**: Before inferring PHIs, double-check whether `my_config_test.train.yaml` already exists. If not, run this:

```bash
awk '{print "  "$0}' results/PHILM_best_params.yaml \
    | cat src/philm/config/config_test.train.yaml - \
    > my_config_test.train.yaml
```

`philm-interaction` is used to infer sensitivity scores for each phage-prokaryote pair:

```bash
philm-interaction -c my_config_test.train.yaml --data-dir data --split train \
    --score-mode gradient --phage-normalize signed_zscore \
    --out results/PHILM_interactions.tsv
```

`results/PHILM_interactions.tsv` has four columns. Among them, `normalized_score` is the final sensitivity score:

```tsv
phage	bacteria	raw_score	normalized_score
vOTU-014862	s__Methanocatella smithii	-0.0072955368	-0.5658288
vOTU-014862	s__Methanoprimaticola hominis	0.011584344	0.88749021
vOTU-014862	s__Bifidobacterium adolescentis	0.011258814	0.86243188
vOTU-014862	s__Bifidobacterium animalis	-0.0023613409	-0.18600857
vOTU-014862	s__Bifidobacterium bifidum	0.0088162394	0.67440945
...
```

**Output:**

* Predicted interaction scores:
  `results/PHILM_interactions.tsv`

### Step 5. Estimate permutation-based P-values for PHIs (Optional)

This optional step estimates empirical p-values and adjusted p-values for PHILM-inferred interactions using permutation-based null models.

For each permutation, the phage profiles are shuffled across samples while the prokaryotic profiles are kept fixed.

* Step 5.1: Retrain the PHILM model based on the permuted data.
* Step 5.2: Calculate the sensitivity scores after model training.
* Step 5.3: Create a random background for each phage using the permutation results.

In this way, we break the sample-level matching between phage profiles and prokaryotic profiles. The random background includes all random scores for that phage against all prokaryotes from all permutation runs. We then compare each real phage-prokaryote score with this background to test whether it is higher than expected by chance and calculate a p-value.

This step can be very computationally intensive for large sample size.
**Note**: It is recommended only when formal false discovery rate (FDR) control is required. It is best suited for high-performance computing environments or systems with sufficient CPU resources. For exploratory analyses, users may first apply a heuristic cutoff, such as `normalized_score >= 3.5`, to prioritize candidate PHIs.

To save time, we use 100 instead of 1,000 permutations. To perform 1,000 permutations, change `N=99` to `N=999` when your computational resources allow.

#### Step 5.1. Retrain models using permuted data

```bash
# Create a configuration file for training a model on each permuted dataset
cat src/philm/config/config_train_perm.yaml \
    <(awk '{print "  "$0}' results/PHILM_best_params.yaml) \
    > train_perm.yaml
# Create jobs
N=99
seq 0 $N \
    | awk '{print "philm-permutate -c train_perm.yaml --perm-id "$1}' \
    > jobs.1.txt
```

`jobs.1.txt` contains 100 jobs. Either run the jobs in parallel using multiple CPUs or submit them as multiple jobs on an HPC cluster.

#### Step 5.2. Infer PHIs from permutation-trained models

```bash
# Create configuration files for obtaining sensitivity scores from each model trained on permuted data
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
# Create jobs
> jobs.2.txt
for f in permutation_infer_configs/config_infer_gradient.perm_*.yaml; do
    i=$(basename "$f" .yaml | cut -f2 -d. | cut -f2 -d_)

    echo "philm-interaction \
        -c permutation_infer_configs/config_infer_gradient.perm_${i}.yaml \
        --data-dir data \
        --split train \
        --score-mode gradient \
        --sort-by none \
        --phage-normalize signed_zscore \
        --out results/permutation_null/perm_${i}/PHILM_perm_${i}.raw_gradient.tsv" \
        >> jobs.2.txt
done
```

`jobs.2.txt` also contains 100 jobs. Either run the jobs in parallel using multiple CPUs or submit them as multiple jobs on an HPC cluster.

#### Step 5.3. Calculate empirical p-values and adjusted p-values

`philm-pval` is used to calculate p-values by comparing observed PHI scores and background scores.

```bash
philm-pval --norm_mode off --observed results/PHILM_interactions.tsv \
    --perm_glob "results/permutation_null/perm_*/PHILM_perm_*.raw_gradient.tsv" \
    --out results/PHILM_interactions.pvalues.tsv
# Save only significant PHIs using adjusted p-value < 0.05
awk -F"\t" 'NR==1 || $5<0.05' \
    results/PHILM_interactions.pvalues.tsv \
    | cut -f1,2,3,5 \
    > results/PHILM_interactions.pvalues.filtered.tsv
```

`results/PHILM_interactions.pvalues.tsv` has six columns. Among them, `padj_BH_normalized_score` is the adjusted p-value for FDR control:

```tsv
phage	bacteria	normalized_score	pval_normalized_score	padj_BH_normalized_score	n_null_normalized_score
vOTU-000001	s__Methanocatella smithii	-0.12759849	0.549245606274343	0.963276900989163	694000
vOTU-000001	s__Methanoprimaticola hominis	1.4148067	0.0793889345980769	0.963276900989163	694000
vOTU-000001	s__Bifidobacterium adolescentis	-0.34392393	0.633610038025882	0.963276900989163	694000
vOTU-000001	s__Bifidobacterium animalis	-0.43413371	0.666925551980473	0.963276900989163	694000
vOTU-000001	s__Bifidobacterium bifidum	0.62822294	0.263522675039373	0.963276900989163	694000
vOTU-000001	s__Bifidobacterium catenulatum	0.27397364	0.389766008982696	0.963276900989163	694000
...
```

**Output:**

* Predicted interaction scores with p-values and adjusted p-values:
  `results/PHILM_interactions.pvalues.tsv`
* Filtered predicted interaction scores:
  `results/PHILM_interactions.pvalues.filtered.tsv`

## Results
Some steps in this tutorial can be time-consuming. To allow users to run individual steps without completing all preceding steps, we provide the essential intermediate results on [Zenodo](https://zenodo.org/records/21269560). Use the following commands to decompress each 7z archive: 
```bash
wget -t 3 -O <compressed_file>.7z https://zenodo.org/records/21269560/files/<compressed_file>.7z
7zz x <compressed_file>.7z
```

The provided archives include:
* `PHILM_input_data.7z`: Contains the input files required by PHILM. These files should be placed in the `data/` directory.
* `PHILM_model_results.7z`: Contains the trained model and prediction metrics, including `PHILM_best_model.pth`, `PHILM_best_params.yaml`, `PHILM_predict_test.ft.metrics`, `PHILM_predict_val.ft.metrics`, and `PHILM_predict_train.ft.metrics`.
* `PHILM_interactions.7z`: Contains the inferred interaction results, including `PHILM_interactions.tsv`, `PHILM_interactions.pvalues.tsv`, and `PHILM_interactions.pvalues.filtered.tsv`.
* `perm_<start number-end number>_scores.tsv.7z`: * `perm_<start-end>_scores.tsv.7z`: These archives contain the permutation-derived PHILM scores used for empirical p-value calculation. Because storing all 1,000 permutation results in a single file would be very large, the results were divided into 50 compressed files. After decompressing all `.7z` files, organize the permutation results into the required PHILM directory structure by running: `python organize_perm_files.py --input "perm_*-*_scores.tsv" --outdir permutation_null`. This command will generate files with the following structure: `permutation_null/perm_*/PHILM_perm_*.raw_gradient.tsv`. Before running Step 5.3, move or place the generated `permutation_null/` directory under the `results/` directory.
