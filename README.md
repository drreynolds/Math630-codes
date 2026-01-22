# Math630-codes

Codes for in-class collaboration for the course: Numerical Linear Algebra (Math 630) at the [University of Maryland Baltimore County](https://www.umbc.edu).

Codes are provided in Matlab, Python, and even some in C++.  Students are expected to write their own programs for the course in either Matlab or Python, and are free to alternate between languages at will throughout the semester.  The C++ codes are only provided as an example of how these would look in a compiled language, and to provide runtime comparisons against interpreted languages like Matlab and Python.  

Software dependencies:
* Matlab: all versions should work for these codes, although I recommend R2015b or newer so that loops can be properly optimized by the just-in-time compiler.
* Python: these require Python version 3+, along with both the Numpy and SciPy packages (SciPy v0.7 or higher).  For students who prefer to set up virtual environments to encapsulate codes for a class, you may use the top-level `python_requirements.txt` file to do so.  For example, to create a virtual environment in the hidden `.venv`    folder at the top-level folder for this cloned repository:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r python_requirements.txt
  ```

  After this installation is complete, you can "test" the installation by running

  ```bash
  python3 -c 'import numpy; import scipy'
  ```

  if this doesn't throw any error messages, then these packages were installed correctly into your environment.  You may "deactivate" this Python environment from your current shell at any time with the command

  ```bash
  deactivate
  ```

  and in the future you can "reactivate" the python environment in your shell by running from the top-level directory of this repository

  ```bash
  source .venv/bin/activate
  ```
  
* C++: these require a compiler that supports the C++11 standard, and the [CMake](https://cmake.org/download/) build system (version 3.10 or higher).

[Daniel R. Reynolds](https://drreynolds.github.io/)  
[Mathematics & Statistics @ UMBC](https://www.mathstat.umbc.edu)
