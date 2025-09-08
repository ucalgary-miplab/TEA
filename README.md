<div align="center">

# Evaluating causal generative models with synthetic ground truth medical image counterfactuals

</div>

This repository contains code associated with the paper:
> E. A. M. Stanley*, V. Vigneshwaran*, E. Y. Ohara*, F. G. Vamosi, N. D. Forkert, M. Wilms. <i>Synthetic Ground Truth Counterfactuals for Comprehensive Evaluation of Causal Generative Models in Medical Imaging</i>. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025.

*<sub><sup>Contributed equally</sub></sup>

## Usage

### Installations 

1. **Install Miniconda/Conda**
2. **Create a new Conda environment:**
   ```bash
   conda create -n tea python=3.10
   
   conda activate tea
   
   pip install -r requirements.txt

### Data directory

The directory should have the following structure:

  ```
   Experiment_name (e.g., exp205)
  ├── data
  |   |- train.csv
  |   |- test.csv
  |   |- val.csv
  |   ├── train
  │   ├── test
  │   └── val

  ```

### Run

**After creating these directories, try running the following command to verify:**

```bash
./run.sh exp205
```

## Data
The datasets used in this paper are available at [doi.org/10.5683/SP3/MQWWW5](https://doi.org/10.5683/SP3/MQWWW5).

## Citation 

