# CounterBias

Counterfactual Evaluation With SimBA

# Installation Instructions

## Installations

1. **Install Miniconda/Conda**
2. **Create a new Conda environment:**
   ```bash
   conda create -n counterbias python=3.10
   
   conda activate counterbias
   
   pip install -r requirements.txt

## Data directory

The directory should have the following structure:

  ```
   Experiment_name (e.g., far_bias)
  ├── data
  |   |- csv
  |   |   |- train.csv
  |   |   |- test.csv
  |   |   |- val.csv
  |   |- pca
  |   |- images
  │      ├── train
  │      ├── test
  │      └── val
  ├── cfs
  └── models
         |- MACAW
         |- SFCN
         |- HVAE
         
  ```

## Run

**After creating these directories, try running the following command to verify:**

```bash
./run.sh far_bias
