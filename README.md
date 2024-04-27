# scikit_raman

scikit_raman is a Python library designed for manipulation and analysis of Raman Spectral Data. This project originated from the author's thesis during their Master's degree in Computer Science at the University of Milan-Bicocca.

## Getting Started

### Installation

#### Prerequisites

Before installing scikit_raman, ensure you have the following prerequisites installed:

- Python 3.x
- Git (for cloning the repository)

#### Installation Steps

To keep your project environment clean and isolated, it's recommended to install scikit_raman within a virtual environment. Here's how to do it:

1. Create a virtual environment for your project:

   ```bash
   python3 -m venv myenv
   ```

   Replace `myenv` with the desired name for your virtual environment.

2. Activate the virtual environment:

   - On Windows:
   
     ```bash
     myenv\Scripts\activate
     ```

   - On Unix or MacOS:
   
     ```bash
     source myenv/bin/activate
     ```

3. Clone the scikit_raman repository:

   ```bash
   git clone https://gitlab.com/marcoplaza98/scikit_raman.git
   ```

4. Navigate into the cloned repository directory:

   ```bash
   cd scikit_raman
   ``` 

5. Install scikit_raman using pip:

   ```bash
   pip install .
   ```

   This will install scikit_raman along with its dependencies into your virtual environment.

6. You're all set! You can now start using scikit_raman within your project.

   Whenever you're finished working with scikit_raman, you can deactivate the virtual environment by running:

   ```bash
   deactivate
   ```

### Usage

Once installed, you can start using scikit_raman in your Python projects. Import the necessary modules and functions as needed:

```python
import scikit_raman
```

### Supported Platforms

scikit_raman should work on any platform where Python is supported. The installation steps provided above are applicable to most Unix-like systems (Linux, macOS). For Windows users, you may need to adjust the commands slightly to accommodate differences in command line interfaces.

### Author

- **Name:** Marco Piazza
- **GitLab:** [marcoplaza98](https://gitlab.com/marcoplaza98)
