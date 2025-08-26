# MSc-Thesis-air-pollution

## Problem Description

The goal of this work is to demonstrate that inverse problem of full-field spatial reconstruction of Air Pollution fields from sparse, movable observations is achieved through data-driven Deep Learning frameworks. 

## Methodology

There are 3 frameworks utilized. The discriminative models include a) ViT+Unet b) Unet. The generative model includes score-based conditional diffusion model.

## How to run the code

1. Clone the repository: **`git clone https://github.com/abhishek30-ml/MSc-Thesis-air-pollution.git`**
2. Place the simulation data in **`data/`** directory.
3. Place the observation data in **`data/new_observation_data/`** directory.
4. Model parameters can be changed in **`config/train_model.yaml`** directory.

### Training the models

1. **`python3 main.py --model=diffusion `** to train a new model. Set model flag to be either of 'vit_unet', 'unet', 'diffusion' to train the respective models.
2. **`python3 main.py --model=diffusion --load_checkpoint`** to retrain the model with existing weights. The trained weights can be found in **`model_weights/`** directory.

### Model inference

1. **`inference/observation_result.ipynb`** notebook depicts the inference results on observation data
2. **`inference/simulation_result.ipynb`** notebook depicts the inference results on validation set of simulation data

## Results

**`result/observation_data/`** and **`result/simulation_data/`** contain the results for respective data types.

* **`output_metric`** in simulation results contains the SSIM measure and mean relative error for various models
* **`relative_error`** contains the numpy files of errors and histogram plots. 

