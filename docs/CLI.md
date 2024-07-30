# CLI Usage Guide

## Overview
The Command Line Interface (CLI) for the scikit_raman library allows you to manage and process Raman spectral data directly from the terminal.

## Commands

### Processing data
Processes Raman spectral data using the specified configuration file and experiment type.

#### Syntax
```bash
scikit-raman-cli --configuration-file <path_to_config> --experiment-type <experiment_type>
```

#### Options

- `--configuration-file`, `-p` (required): Path to the configuration YAML file.
- `--experiment-type`, `-t` (required): Type of the experiment. Must be either `dl` or `ml`.

#### Examples

##### Process a DL Experiment

```bash
scikit-raman-cli -p dl_config.yaml -t dl
```

##### Process a ML Experiment

```bash
scikit-raman-cli -p ml_config.yaml -t ml
```