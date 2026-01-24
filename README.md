# IDSpace
IDSpace is a synthetic data generation framework to generate a large number of identity documents using only a few documents from a target domain without including any private information.

#### Avaliable Synthetic Dataset
Our synthetic dataset is released on HuggingFace, you can download the datasets from [here](https://huggingface.co/datasets/Anonymous-111/IDSPACE).

## Environment Setup and Installations
Python>=3.10 is required to run the project. To install all the dependencies, either run the following command or manually install the dependencies in the [requirements](/requirements.txt) file.
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

### Template images generation process

See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/template_image) in template_image for more details.


### Scanned Images generation process
See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/scanned_image) in scanned_image for more details.

### Mobile images generation process
See [README.md](https://github.com/asu-cactus/IDSpace/tree/master/mobile_image) in mobile_image for more details.


## An Example of Generating a Slovakia Document Using IDSpace

### Step 1: Download the Repository and environment setup

```bash
git clone https://github.com/asu-cactus/IDSpace.git
cd IDSpace
pip install -r requirements.txt # python >=3.10 is required
```

### Step 2: Customize the configuration file used for BO search and for template image generation.
#### 2.1: Go to the folder
```bash
cd data/inputs/SVK
```
In this folder, SVK_configure.yaml contains all configuration settings used for both the Bayesian Optimization (BO) search and template-based image generation. The file SVK_pii_configure.yaml defines user-configurable parameters for generating personal data according to specified distributions.

#### 2.2: Customize SVK_configure.yaml
Example Format
- Defines experiment-level settings
```yaml
area: SVK                           # Country / document area identifier
template_path: data/inputs/SVK/template_SVK.png   # Base ID template image

# Font resources
text_fonts_path: data/inputs/text_fonts/          # Fonts used for printed text
signature_fonts: data/inputs/signature_fonts      # Fonts used for signatures
```
- Defines personal metadata
  
We support two input formats. A CSV file can be provided to supply user-defined personal metadata for populating each template field. Alternatively, a YAML file can be used to specify metadata distributions, in which case the system automatically generates personal metadata that satisfies the specified requirements.
```yaml
personal_info: data/inputs/SVK/personal_info.csv  # CSV containing personal information
# personal_info: data/inputs/SVK/SVK_pii_configure.yaml  # yaml containing personal info distribution
```
- Specifies Bayesian Optimization settings
```yaml
bo_settings:
  init_points: 1       # Number of random initialization points
  n_iter: 3            # Number of optimization iterations
  seed: 2              # Random seed for reproducibility
```
- Specifies Objective configuration
```yaml
eval_args:
  target_samples: 1          # Number of samples used for parameter search
  with_model: 1              # Whether to use the model-guided method
  candidate_models:          # Guided Models 
    - resnet50
  testing: true              # Enable testing in the end
  lambda0: 1                 # scoring weight for SSIM
  lambda1: 1                 # scoring weight for model consistency
  guided_datapaths: data/inputs/template_guided_datas.json # Guided data info
```

- Defines output directories
```yaml
output:
  best_settings_file: data/outputs/SVK/template_best_settings.json
  synthetic_images_path: data/outputs/SVK/positive  # Output directory for generated images
  synthetic_images_annotation_path: data/outputs/SVK/SVK_original_annotation.json #output annotation file
```
- Define template segments
```yaml
segments:

  portrait:
    type: image              # Image-based segment
    bbox:                    # Bounding box (x1, y1, x2, y2)
      - 44
      - 176
      - 370
      - 576
    tunable: false           # Fixed placement

  # ---------- Text fields ----------
  name:
    type: text
    font_group: 1            # Font style group
    color_group: 1           # Color group reference
    tunable: true            # Optimized during BO search
    font_info:
      text_height:
        initial: 29          # initial value/ approximation
        margin: 5            # Allowed variation
      text_position:
        x:
          initial: 397
          margin: 5
        y:
          initial: 211
          margin: 5
```
#### 2.3: Customize SVK_pii_configure.yaml
If a user does not have personal metadata, or wants to generate images with demographic attributes following a specific distribution, they can customize the SVK_pii_configure.yaml file, and the system will automatically generate personal metadata that satisfies the specified requirements.

Example Format
- number of generated examples and used modules
```yaml
number_samples: 5

ui:
  default_modules: ["person_core", "dob", "doc_dates", "issue_location", "svk_numbers", "portrait_from_index"]
  default_profile: "SVK"
  default_seed: 42
```
- output fields, keys, and formats
```yaml
output_profiles:
  SVK:
    description: "SVK-style output format"
    fields:
      - last_name
      - first_name
      - dob
      - issue_date
      - expiry_date
      - gender
      - issue_place
      - doc_number
      - local_id_number
      - signature
      - portrait

    key_map:
      last_name: surname
      first_name: name
      dob: birth_date
      doc_number: number
      local_id_number: id_number

    formats:
      date_pattern: "%d.%m.%Y"
      gender_encoding:
        male: "M"
        female: "F"
        nonbinary: "X"
```
- Distribution of one attribute
```yaml
modules:
  dob:
    description: "Generate integer age and DOB string."
    provides: ["age", "dob_iso", "dob"]
    requires: []
    params:
      age:
        dist:
          type: uniform_int
          params:
            min: 18
            max: 85
          enforce: sample
```

### Step 3: Run the pipeline
Run the pipeline:

To run the Bayesian Optimization (BO) search, a customized YAML configuration file must be provided.

To perform image generation, the best-performing parameter configuration produced by the BO search is used.
```bash
cd ../../../ # go th the main folder
python template_image/template_image/Optuna_search_global.py --config data/inputs/SVK/SVK_configure.yaml #For BO parameters search
python template_image/template_image_generation.py --best_seting_path data/outputs/SVK/template_best_settings.json #For template images generation
```
### Step 4: Check the Output
Output Directory
Defined in SVK_configure.yaml:
```yaml
output:
  best_settings_file: data/outputs/SVK/template_best_settings.json
  synthetic_images_path: data/outputs/SVK/positive  # Output directory for generated images
  synthetic_images_annotation_path: data/outputs/SVK/SVK_original_annotation.json #output annotation file
```
Example Output Structure
```bash
data/outputs/SVK/
├── positive/
│   ├── generated_1.png
│   └── generated_2.png
├── SVK_original_annotation.json
├── template_best_settings.json
```


