# scikit_raman

scikit_raman is a Python library designed for manipulation and analysis of Raman Spectral Data. This project originated from the author's thesis during their Master's degree in Computer Science at the University of Milan-Bicocca.

## Installation

### Prerequisites

Before installing scikit_raman, ensure you have the following prerequisites installed:

- Python 3.9 or higher
- Pip version 24.0 or higher
- Git (for cloning the repository)

### Installation Steps

To keep your project environment clean and isolated, it's recommended to install scikit_raman within a virtual environment.

#### Standard Installation (For Users):

If you only intend to use the library:

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
   git clone https://gitlab.com/marcoplaza98/scikit_raman.git
   ```

4. Navigate into the cloned repository directory:

   ```
   cd scikit_raman
   ``` 

5. Install scikit_raman using pip:

   ```
   pip install .[installation_option]
   ```

   This will install scikit_raman along with its dependencies into your virtual environment.

   Replace `installation_option` with `standardKeras`, `mindHardKeras` or `torch` depending on your desired installation.

   **Note**: For a correct usage of the library is fondumental to set one of the two installation options!!!

   For more details, please refer to the [Tensorflow Documentation](https://www.tensorflow.org/install/source?hl=en#gpu)

#### Editable Installation (For Developers):

If you intend to contribute to the library or need to work on it:

1. Follow steps 1 to 4 from the standard installation instructions above.

2. Install scikit_raman in editable mode using pip:

   ```
   pip install -e .[installation_option]
   ```

   Replace `installation_option` with `standardKeras`, `mindHardKeras` or `torch` depending on your desired installation.

   **Note**: For a correct usage of the library is fondumental to set one of the two installation options!!!

   For more details, please refer to the [Tensorflow Documentation](https://www.tensorflow.org/install/source?hl=en#gpu)

   If installation in editable mode fails, ensure your setuptools version is up to date. You can upgrade it with the following commands:

   ```
   pip install --upgrade setuptools
   ```
   
You're all set! You can now start using scikit_raman within your project.

Whenever you're finished working with scikit_raman, you can deactivate the virtual environment by running:

```
deactivate
```

## Quick Start

#### Using the Python API

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
#### Using the CLI

```bash
# Display help
scikit-raman-cli --help

# Process a DL experiment using a configuration file
scikit-raman-cli -p dl_config.yaml -t dl

# Process an ML experiment using a configuration file
scikit-raman-cli -p ml_config.yaml -t ml
```
### Configuration

The library uses a main `config.yaml` file that references additional configuration files. For detailed information on all avaiable options and how to structure these files, please refer to [CONFIGURATION.md](docs/CONFIGURATIONS.md)

### CLI Usage

The CLI provides commands for processing spectral data and managing configurations. For detailed CLI usage, see [CLI.md](docs/CLI.md)

### Supported Platforms

scikit_raman should work on any platform where Python is supported. The installation steps provided above are applicable to most Unix-like systems (Linux, macOS). For Windows users, you may need to adjust the commands slightly to accommodate differences in command line interfaces.

### Contributors

- **Name:** Riccardo Frigerio
- **GitHub:** [RFrig16](https://github.com/RFrig16)

### Author

- **Name:** Marco Piazza
- **GitHub:** [piazzam](https://github.com/piazzam)
