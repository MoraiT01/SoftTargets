"""Install all needed packages for the project and downloads the datasets."""
from setuptools import find_packages, setup
from pathlib import Path

# Function to read the contents of the requirements.txt file
def read_requirements(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith(('#', '-'))
    ]

# Path to your local requirements file
REQUIREMENTS_PATH = 'requirements.txt'
install_requirements = read_requirements(REQUIREMENTS_PATH)

setup(
    name='softtarget_MU',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requirements, # Use the list generated from requirements.txt
    # ... rest of your setup metadata ...
    author='Moritz Kastler',
    author_email='moritz.kastler@outlook.de',
    description='This project aims to show the differences between using hard and soft labels' \
    'for Machine Unlearning Algorithms, like Gradient Ascent, Gradient Difference, RMU '
    'and Tarun, Ayush K., et al. "Fast yet Efficient Unlearning Machine Unlearning"',
    url='https://github.com/MoraiT01/SoftTargets',
)