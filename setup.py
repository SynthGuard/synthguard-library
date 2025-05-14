import setuptools
import subprocess
import sys
import time

# First install torch with the custom index URL
print('Installing torch...')
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
print('Torch installation completed...')

# Sleep for 10 seconds after installing torch
time.sleep(10)

# Now, install other dependencies, starting with sdv
# to_be_installed = [
#     'sdv==1.17.1',
# ]

# Now, install other dependencies, starting with sdv
to_be_installed = [
    'sdv==1.17.1',
    'matplotlib==3.7.3',
    'pandas',
    'synthgauge',
    'catboost',
    'xmlschema'
]


# Install each package from the list
for package in to_be_installed:
    print(f'Installing {package}...')
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f'{package} installation completed...')

# Now proceed with the rest of the setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthguard",
    version="0.0.1",
    author="ktamm",
    author_email="kristian.tamm@cyber.ee",
    description="Data Synthesis Pipeline Module Library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://gitlab.cyber.ee/exai/synthguard",
    packages=setuptools.find_packages(),
    install_requires=[
        # Other dependencies if necessary
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='synthetic data, privacy, machine learning',
)
