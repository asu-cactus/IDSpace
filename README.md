# IDSpace
IDSpace is a synthetic data generation framework to generate a large number of identity documents using only a few documents from a target domain without including any private information.

## Environment Setup and Installations
Python>=10.0 is required to run the project. To install all the dependencies, either run the following command or manually install the dependencies in the [requirements](/requirements.txt) file.
```bash
pip install -r requirements.txt
```


## Setting Up Datasets and Models
#### Setting Up Models
Download the pretrained models from [here](https://huggingface.co/datasets/cactuslab/IDSpace/blob/main/models.tar.gz) and place the unzipped models folder inside the data/ directory.

#### Setting Up Target Domain Images
Download the target domain images file from [here](https://drive.google.com/file/d/1iqZ0rDuO0GSkc3Osrr7V_--JbvPOky5X/view?usp=sharing) and place the unzipped 'target_images' folder inside the data/ directory.

#### Setting Up Synthetic Dataset
Download the datasets from [here](https://huggingface.co/datasets/cactuslab/IDSpace) and place the unzipped folders inside the data/ directory.


## Running the Experiments Scripts

### Running Bayesian Optimization
In this Experiments, we used SIDTD tmplate dataset as our target domain and used in our experiments, you can download the SIDTD data [here](https://github.com/Oriolrt/SIDTD_Dataset/tree/main/SIDTD/data/DataLoader#run-example) or [here](https://tc11.cvc.uab.es/datasets/SIDTD_1/), then put the `reals` and 'fakes' folders inside the data/templates/ directory. 
To run the Bayesian Optimization baseline, run the following command under the root directory of the project.
```bash
python experiments/Bayesian_search.py target_samples with_model lambda0 lambda1 candidate_models
```
In the above command, the parameter target_samples denotes the number of samples to be used (int), with_model denotes whether the optimization will be model-guided or not (0 or 1), lambda0 and lambda1 parameters control the fractions of similarity score and consistency score in the total evaluation score, and candidate_models stand for the names of models (space separated) that will guide the optimization. Example commands are given below:
##### Example of Search W/ Model-Guided Optimization
```bash
python experiments/Bayesian_search.py 20 1 1 1 resnet50
```
##### Example of Search W/ SSIM Only Objective
```bash
python experiments/Bayesian_search.py 20 0 1 1 ssim
```
You may also execute experiments/run.py which provides sample commands to execute Bayesian_search.py.


### Running Hyperband Search Baseline
To run the Hyperband search baseline, run the following command under the root directory of the project.
```bash
python experiments/Hyperband_search.py param_r param_eta target_samples with_model lambda0 lambda1 candidate_models
```
Here, param_r and param_eta represent maximum resources and successive halving parameters of the Hyperband search method. Other parameters target_samples, with_model, lambda0, lambda1, and candidate_models are similar to those in Bayesian search. Example commands:
##### Example of Search W/ Model-Guided Optimization
```bash
python experiments/Hyperband_search.py 700 3 20 1 1 1 resnet50
```
##### Example of Search W/ SSIM Only Objective
```bash
python experiments/Hyperband_search.py 700 3 20 0 1 1 ssim
```
Also, experiments/run_hyperband.py can be executed, which runs Hyperband_search.py with example commands.


### Running CycleGAN Baseline
Download the repo of CycleGAN [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git)
Download the data used for this baseline [here](https://drive.google.com/file/d/1tPArY3u3PpSW8IgWtqjnASymyb-iEgFR/view?usp=sharing)

To run the CycleGAN baseline, run the following command under the root directory of the project.
```bash
cd experiments
bash cycle_run.sh
```
This include the CycleGAN training part and results evaluation part(test_cyclegan.py)


### Running Bayesian Optimization on Scanned Images
see [here](https://github.com/asu-cactus/IDSpace/tree/master/scanned_image#bayesian-optimization-for-scanned-image-processing) for more details.


### Running Bayesian Optimization on Template Images
see [here](https://github.com/asu-cactus/IDSpace/tree/master/template_image#bayesian-optimization-for-template-image-processing) for more details.
