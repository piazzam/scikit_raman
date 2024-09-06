# scikit_raman

scikit_raman is a Python library designed for manipulation and analysis of Raman Spectral Data. This project originated from the author's thesis during their Master's degree in Computer Science at the University of Milan-Bicocca.

## 📌 Table of Contents

- [scikit\_raman](#scikit_raman)
  - [📌 Table of Contents](#-table-of-contents)
  - [🔗 Dependencies](#-dependencies)
  - [🛠️ Configuration](#️-configuration)
  - [📖 Usage](#-usage)
    - [Python API](#python-api)
    - [CLI](#cli)
  - [🌎 Supported Platforms](#-supported-platforms)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🔗 Dependencies

Before installing scikit_raman, ensure you have the following prerequisites installed:

- [Git](https://git-scm.com): Required for cloning the repository.
- [Python 3.9+](https://www.python.org): Make sure you have Python version 3.9 or higher.
- [pip](https://pypi.org/project/pip/): Have the latest version of pip installed.
- [setuptools](https://pypi.org/project/setuptools/): Have the latest version of setuptools installed.

## 🛠️ Configuration

The library uses a main `config.yaml` file that references additional configuration files. For detailed information on all avaiable options and how to structure these files, please refer to [CONFIGURATION.md](docs/CONFIGURATIONS.md)

## 📖 Usage

### Python API

```python
import scikit_raman.Classes.Experiment.DLExperiment as DLE
import scikit_raman.Classes.Experiment.MLExperiment as MLE

# Load and run a DL experiment
dl_experiment = DLE.DLExperiment('dl_config.yaml')
dl_experiment.experiment()

# Load and run an ML experiment
ml_experiment = MLE.MLExperiment('ml_config.yaml')
ml_experiment.experiment()
```
### CLI

The CLI provides commands for processing spectral data and managing configurations. For detailed CLI usage, see [CLI.md](docs/CLI.md)

```bash
# Display help
scikit-raman-cli --help

# Process a DL experiment using a configuration file
scikit-raman-cli -p dl_config.yaml -t dl

# Process an ML experiment using a configuration file
scikit-raman-cli -p ml_config.yaml -t ml
```

## 🌎 Supported Platforms

scikit_raman should work on any platform where Python is supported. The installation steps provided above are applicable to most Unix-like systems (Linux, macOS). For Windows users, you may need to adjust the commands slightly to accommodate differences in command line interfaces.

## 🙏 Acknowledgments

We extend our gratitude to the following individuals for their contributions to this project:

- **Marco Piazza** ([piazzam](https://github.com/piazzam))
- **Riccardo Frigerio** ([RFrig16](https://github.com/RFrig16))
