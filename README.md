# IDSpace
IDSpace is a synthetic data generation framework to generate a large number of identity documents using only a few documents from a target domain without including any private information.

#### Avaliable Synthetic Dataset
Our synthetic dataset is released on HuggingFace, you can download the datasets from [here](https://huggingface.co/datasets/Anonymous-111/IDSPACE).

## Environment Setup and Installations
Python>=10.0 is required to run the project. To install all the dependencies, either run the following command or manually install the dependencies in the [requirements](/requirements.txt) file.
```bash
pip install -r requirements.txt
```


## Setting Up Datasets and Models
#### Setting Up Models
Download the pretrained models from [here](https://huggingface.co/datasets/Anonymous-111/IDSPACE/blob/main/models.tar.gz) and place the unzipped models folder inside the data/ directory.

#### Setting Up Target Domain Images
In this Experiments, we used SIDTD tmplate dataset as our target domain and used in our experiments, you can download the SIDTD data in [site1](https://github.com/Oriolrt/SIDTD_Dataset/tree/main/SIDTD/data/DataLoader#run-example) or [site2](https://tc11.cvc.uab.es/datasets/SIDTD_1/), then put the `reals` and `fakes` folders inside the data/templates/ directory. 


## Running the Experiments Scripts

### Running Bayesian Optimization
To run the Bayesian Optimization baseline, run the following command under the root directory of the project.
```bash
python experiments/Bayesian_search.py target_samples with_model lambda0 lambda1 candidate_models
```
In the above command, the parameter target_samples denotes the number of samples to be used (int), with_model denotes whether the optimization will be model-guided or not (0 or 1), lambda0 and lambda1 parameters control the fractions of similarity score and consistency score in the total evaluation score, and candidate_models stand for the names of models (space separated) that will guide the optimization.
Example commands are given below:
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
This include the CycleGAN training part and results evaluation part(test_cyclegan.py).

## An Example of Generating a Slovakia Document Using IDSpace

See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/template_image) in template_image for more details.

We provide a configuration example on the identity document from SVK.

In data/inputs/SVK folder:

SVK_configure.yaml contains all the configurations used for BO search and for template image generation.

SVK_pii_configure.yaml contains parameters user can define if the user wants to generate some personal data based on a distribution.

To run the pipeline:
```bash
python template_image/Opcd experiments
template_image/Optuna_search_global.py --config data/inputs/SVK/SVK_configure.yaml #For BO parameters search
python template_image/template_image_generation.py --best_seting_path data/outputs/SVK/template_best_settings.json #For template images generation
```


### Running Bayesian Optimization on Scanned Images
See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/scanned_image) in scanned_image for more details.

### Mobile images generation process
See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/mobile_image) in mobile_image for more details.
