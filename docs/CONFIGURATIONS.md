# Configuration Guide

## Overview
This document provides detailed information about the configuration options available for the spectral data management library. The configuration is split across multiple YAML files for better organization.

**NOTE:** It's important to define all the configurations file inside the same folder!!!

## Main Configuration File (`config.yaml`)

### Description
The `config.yaml` file is the main configuration file that references other configuration files for more specific settings.

### Options

- **file_name** (string): Path to the dataset file.
- **seed** (int): Random seed for reproducibility.
- **label_dictionary** (dict): Mapping of labels to numeric values.
  - Example:
    ```yaml
    label_dictionary: 
    {
      'ASMA':0,
      'BPCO':1}
    ```
- **with_drugs** (bool): Whether to include drug data.
- **drug_names** (list): List of drug names.
  - Example: `[ASMA, BPCO]`
- **polvere** (bool): Whether to check if drugs are in solid (polvere) or fluid (in_fisio) form.
- **farmaci_files** (list): List of paths to CSV files containing drug informations about patients.
  - Example:
    ```yaml
    farmaci_files: [../new_dataset/therapy_bpco.csv, ../new_dataset/asma_farmaci.csv]
    ```
- **farmaci_ds_filename** (string): Path to the drug spectral dataset.
- **label_dictionary_drugs** (dict): Mapping of drug labels to numeric values.
  - Example:
    ```yaml
    label_dictionary_drugs:
    {
      polvere: 0,
      in_fisio: 1}
    ```
- **with_preprocessing** (bool): Whether to apply preprocessing.
- **preprocessing_file** (string): Path to the preprocessing configuration file. Default is `preprocessing.yaml`. Click [here](#preprocessing-configuration-file-preprocessingyaml) for more details.
- **with_pca** (bool): Whether to apply PCA. 
- **pca_components** (int): Number of PCA components if PCA is applied.
- **model_name** (string): Name of the model. The type of experiment determines the available options.
  - **dl (Deep Learning)**:
    - If `model_name` is `benchmark`, the benchmark model developed in the library is used. It is also necessary to define the `n_dims` option of the model, which is critical for correctly defining the model in order to respect the number of points (or length) of the spectral data.
      - Example:
        ```yaml
        model_name: benchmark
        n_dims: 991
        ```
    - Otherwise, the user must provide the path to a saved model in the `model_path` option. Additionally, the `model_configuration` option must be selected, and the path to the YAML model parameters file must be entered. Click [here](#model-parameters-file-model_configurationyaml) for more details.
      - Example:
        ```yaml
        model_path: ../my_models/dl_model.keras
        model_configuration: model.yaml
        ```
  - **ml (Machine Learning)**:
    - The `model_name` option may be populated with `xgboost`, `lda`, or `svm`, which are all models defined within the library.
      - Example:
        ```yaml
        model_name: xgboost
        ```
    - If other models are desired, the user is required to import the selected model by providing the appropriate path information in the `model_path` option.
      - Example:
        ```yaml
        model_path: ../my_models/ml_model.pkl
        ```
- **evaluation_procedure** (string): Type of cross-validation. There are two options: `k_fold` and `loocv`.
  - **k_fold**: Executes a standard k-fold cross-validation.
    - **k_fold_parameter_file** (string): Path to the k-fold parameters file. Click [here](#k-fold-parameters-file-k_foldyaml) for more details.
    - Example:
      ```yaml
      evaluation_procedure: k_fold
      k_fold_parameter_file: k_fold.yaml
      ```
  - **loocv**: Refers to Leave One Out Cross Validation, a type of cross-validation developed exclusively for this type of analysis. For each iteration, it considers a single patient with all of his spectral data as the test set. All other patients and spectra are used in the training set.
    - **loocv_parameter_file** (string): Path to the loocv parameters file. Click [here](#loocv-parameters-file-loocvyaml) for more details.
    - Example:
      ```yaml
      evaluation_procedure: loocv
      loocv_parameter_file: loocv.yaml
      ```
- **n_classes** (int): Number of classes. It is a mandatory input parameter for both cross-validation types and the benchmark model if it is selected.
- **path_out** (string): Output directory path.
- **evaluator_file** (string): Path to the evaluator configuration file. Click [here](#evaluator-configuration-file-evaluatoryaml) for more details.
- **classes** (list): List of class labels.
  - Example: `[0, 1]`
- **target_names** (list): List of target names.
  - Example: `[ASMA, BPCO]`

### Example
```yaml
file_name: ../new_dataset/pickle_old/data_asma_bpco.pkl
seed: 42
label_dictionary:
  {
    'ASMA':0,
    'BPCO':1}
with_drugs: True
drug_names: [ASMA, BPCO]
polvere: True
farmaci_files: [../new_dataset/therapy_bpco.csv, ../new_dataset/asma_farmaci.csv]
farmaci_ds_filename: ../new_dataset/pickle_old/farmaci_tipo_soluzione.pkl
label_dictionary_drugs:
  {
    polvere: 0,
    in_fisio: 1}
with_preprocessing: True
preprocessing_file: preprocessing.yaml
with_pca: False
pca_components: 20
model_name: benchmark
n_dims: 991
evaluation_procedure: k_fold
k_fold_parameter_file: k_fold.yaml
n_classes: 2
path_out: results/
evaluator_file: evaluator.yaml
classes: [0, 1]
target_names: [ASMA, BPCO]
```

## Preprocessing Configuration File (`preprocessing.yaml`)

### Description
The `preprocessing.yaml` file handle all desired preprocessing steps and their associated parameters.

### Options

- **preprocessing_steps** (list): List with the names of preprocessing steps to apply. The names should match the preprocessing steps available in the library.
- **parameters** (dict): Dictionary containing an association with function name defined above and its parameters.

### Example

```yaml
preprocessing_steps: [delete_uninformative_spectra, resample_shift]
parameters:{
        resample_shift:{
                        start: 450}
}
```

## Model Parameters File (`model_configuration.yaml`)

### Description
The `model_configuration.yaml` file is responsible for managing the parameters of the selected model when it is imported and is not one of the models already defined in the library.

### Options

### Example

## K-Fold Parameters File (`k_fold.yaml`)

### Description
The `k_fold.yaml` file is used for set all parameters for the K-Fold Cross Validation.

### Options

### Example

## LOOCV Parameters File (`loocv.yaml`)

### Description
The `loocv.yaml` file is used for set all parameters for the Leave One Out Cross Validation.

### Options

### Example

## Evaluator Configuration File (`evaluator.yaml`)

### Description
The `evaluator.yaml` is the configuration file which defines all the desired outputs. You can save confusion matrices, tables with the most relevant metrics and the data itself after manipulation.

### Options

### Example
