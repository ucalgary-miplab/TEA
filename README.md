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
  ├── csv
  ├── images
  │   ├── train
  │   ├── test
  │   └── val
  ├── data (for PCA-encoded)
  ├── cfs
  └── models
  ```
  
## Run
 
**After creating these directories, try running the following command to verify if the encoding process works:**
```bash
python encode.py moin_bias
