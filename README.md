# scikit_raman

scikit_raman is an internal Python library designed for manipulation and analysis of Raman Spectral Data. This project originated from the author's thesis during their Master's degree in Computer Science at the University of Milan-Bicocca.

## 📌 Table of Contents

- [scikit\_raman](#scikit_raman)
  - [📌 Table of Contents](#-table-of-contents)
  - [📚 Documentation](#-documentation)
  - [🔗 Dependencies](#-dependencies)
  - [⚙️ Installation](#️-installation)
    - [Method 1: Standard Installation](#method-1-standard-installation)
    - [Method 2: Editable Installation](#method-2-editable-installation)
    - [Installation options](#installation-options)
  - [🛠️ Configuration](#️-configuration)
  - [📖 Usage](#-usage)
    - [Python API](#python-api)
    - [CLI](#cli)
  - [🌎 Supported Platforms](#-supported-platforms)
  - [📝 Citation](#-citation)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 📚 Documentation

The complete documentation, including configuration options, command-line usage, and API references, is available at:

**[scikit_raman Documentation](https://piazzam.github.io/scikit_raman/)**

## 🔗 Dependencies

Before installing scikit_raman, ensure you have the following prerequisites installed:

- [Git](https://git-scm.com): Required for cloning the repository.
- [Python 3.9+](https://www.python.org): Make sure you have Python version 3.9 or higher.
- [pip](https://pypi.org/project/pip/): Have the latest version of pip installed.
- [setuptools](https://pypi.org/project/setuptools/): Have the latest version of setuptools installed.

## ⚙️ Installation

To keep your project environment clean and isolated, it's recommended to install scikit_raman within a virtual environment.

1. Create a virtual environment for your project:

   ```
   python3 -m venv myenv
   ```

   Replace `myenv` with the desired name for your virtual environment.

2. Activate the virtual environment:

   - On Windows:
   
     ```
     myenv\Scripts\activate
     ```

   - On Unix or MacOS:
   
     ```
     source myenv/bin/activate
     ```

3. Clone the scikit_raman repository:

   ```
   git clone https://github.com/piazzam/scikit_raman.git
   ```

4. Navigate into the cloned repository directory:

   ```
   cd scikit_raman
   ``` 

### Method 1: Standard Installation

   ```
   pip install .[installation_option]
   ```

### Method 2: Editable Installation

   ```
   pip install -e .[installation_option]
   ```
   
### Installation options

   Replace `installation_option` with one of the following depending on your needs:

   - `standardKeras`: This option installs the library and its dependencies for general purposes using Keras with CPU support.
   - `mindHardKeras`: This option installs the specific dependencies required to use the library with the GPU configuration of the mindHard virtual machine (the one in the MIND laboratory).
   - `torch`: For using the PyTorch implementation of the methods instead of the Keras version. 

   If you intend to use the library on your machine with a specific GPU configuration, we recommend referring to the [Tensorflow Documentation](https://www.tensorflow.org/install/source?hl=en#gpu) to understand what you need to install.

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

scikit_raman was primarily developed and used in Linux-based research environments with Python 3.9 and 3.10. Other operating systems and Python versions may work but have not been systematically tested. GPU support depends on the selected deep-learning framework and the compatibility of the local CUDA environment.

## 📝 Citation

If you use or refer to `scikit_raman` in academic work, please cite the publication describing the computational pipeline:

> D. Bertazioli, M. Piazza, C. Carlomagno, A. Gualerzi, M. Bedoni, and E. Messina, “An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples,” *Computers in Biology and Medicine*, vol. 171, article 108028, 2024. https://doi.org/10.1016/j.compbiomed.2024.108028

```bibtex
@article{bertazioli2024integrated,
  title   = {An integrated computational pipeline for machine learning-driven diagnosis based on Raman spectra of saliva samples},
  author  = {Bertazioli, D. and Piazza, M. and Carlomagno, C. and Gualerzi, A. and Bedoni, M. and Messina, E.},
  journal = {Computers in Biology and Medicine},
  volume  = {171},
  pages   = {108028},
  year    = {2024},
  doi     = {10.1016/j.compbiomed.2024.108028}
}
```

## 📄 License

This project is licensed under the BSD 3-Clause License.
See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We extend our gratitude to the following individuals for their contributions to this project:

- **Marco Piazza** ([piazzam](https://github.com/piazzam))
- **Riccardo Frigerio** ([RFrig16](https://github.com/RFrig16))
