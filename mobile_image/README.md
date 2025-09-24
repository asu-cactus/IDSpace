# Installation

```
# Step 1: Create conda environment from env.yml
conda env create -f env.yml

# Step 2: Activate the environment (replace 'env' with the actual env name if different)
conda activate env

# Step 3: Install GroundingDINO and segment-anything
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -q -e .
pip install segment-anything
```

# Data Preparation

Download the MIDV dataset

Fill in the survey at http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html and you will receive a link to download the dataset such as the one below.

```
Server URL: sftp://l3i-share.univ-lr.fr
Login (without the quotes): "userMIDV2020"
Password (without the quotes, including spaces): "xxxxxxxx"
```

To download the dataset, you can use the following command:

```
sftp -oHostKeyAlgorithms=+ssh-rsa userMIDV2020@l3i-share.univ-lr.fr
```
It will prompt for the password, then enter the password provided in the email.
```
xxxxxxxx
```
After logging in, you can use the `get` command to download the dataset. For example, you can use:

```
ls                   
cd MIDV2020/dataset          
get photo.rar
```
Then unzip the downloaded file to a directory of your choice. 


# Attribute/Label generation

## Background Images from MIDV2020

```
python bg_attribute.py \
    --image_dir /path/to/MIDV2020/dataset/photo \
    --output ./background_attribute.json \
```

## IDs

```
python id_attribute.py \
    --image_dir /path/to/ID_dataset \
    --output ./id_attribute.json \
```

## Demon for Mobile Image Generation
```
python demo.py
```

# Label Evaluation



## CLIP Score Evaluation
To evaluate the CLIP score of the generated labels, you can use the following command. 
```
python evaluate_clip_score.py \
    --json_path ./background_attribute.json \
    --image_dir /path/to/MIDV2020/dataset/photo \
```

## BLIP Score Evaluation
To evaluate the BLIP score of the generated labels, you can use the following command. 
```
python evaluate_blip_score.py \
    --json_path ./background_attribute.json \
    --image_dir /path/to/MIDV2020/dataset/photo \
```

## ALBEF Score Evaluation
To evaluate the ALBEF score of the generated labels, you can use the following command. 
```
python evaluate_albef_score.py \
    --json_path ./background_attribute.json \
    --image_dir /path/to/MIDV2020/dataset/photo \
```

## VQA Evaluation
```
python vqa_evaluation.py \
    --json_path ./background_attribute.json \
    --image_dir /path/to/MIDV2020/dataset/photo \
```
