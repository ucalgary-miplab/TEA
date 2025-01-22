# TEA

Evaluating causal generative models with synthetic ground truth medical image counterfactuals.


## Installations

1. **Install Miniconda/Conda**
2. **Create a new Conda environment:**
   ```bash
   conda create -n tea python=3.10
   
   conda activate tea
   
   pip install -r requirements.txt

## Data directory

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

## Run

**After creating these directories, try running the following command to verify:**

```bash
./run.sh exp205
