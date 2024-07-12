# Image Work

This repository contains code for feature extraction, transformation, and evaluation using models such as ResNet and ZFNet.

## Files Overview

- `resnet.py`: Contains the ResNet model implementation.
- `zfnet.py`: Contains the ZFNet model implementation.
- `linear_transform.py`: Implements linear transformation.
- `mmd_transform.py`: Implements MMD transformation.
- `class_linear_transform.py`: Implements class-based linear transformation.
- `test_trans.py`: Main script to run feature extraction, transformation, and evaluation.


## Quick Start: Running the Experiment Script

To run all tests and generate line plots of the results, run the `run_experiments.py` script.


## Running the Main Test Script

To test the feature extraction and transformation process, run the `test_trans.py` script.

### Parameters

- `--transform_method`: Specify the transformation method to use. Options are:
  - `none`: No transformation.
  - `linear`: Apply linear transformation.
  - `class_linear`: Apply class-based linear transformation.
  - `MMD`: Apply Maximum Mean Discrepancy (MMD) transformation.
- `--top_k`: Specify the top-k value for top-k accuracy evaluation (e.g., 5).

### Example Usages


# No transformation with top-k=5
python test_trans.py --transform_method none --top_k 5

# Linear transformation with top-k=5
python test_trans.py --transform_method linear --top_k 5

# Class-based linear transformation with top-k=5
python test_trans.py --transform_method class_linear --top_k 5

# MMD transformation with top-k=5
python test_trans.py --transform_method MMD --top_k 5

# Linear transformation with top-k=10
python test_trans.py --transform_method linear --top_k 10

# MMD transformation with top-k=10
python test_trans.py --transform_method MMD --top_k 10



