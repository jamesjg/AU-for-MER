# AU-for-MER: Continuous Action Units for Micro-Expression Recognition

This repository contains code of ICIP2025:
"CONTINUOUS ACTION UNIT INTENSITY MODELING FOR MICRO-EXPRESSION RECOGNITION".

## Environment Setup
1. Clone the repository:

```bash
git clone git@github.com:jamesjg/AU-for-MER.git
cd AU-for-MER
```
2. Create and activate a conda environment:
```bash
conda create -n au_mer python=3.8
conda activate au_mer
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Preparation

1. Obtain Original Dataset

    Download the original dataset images and corresponding micro-expression annotation files (xlsx format) from the official source.

2. Generate Optical Flow
    Following the approach of [HTnet](https://github.com/wangzhifengharrison/HTNet) to generate and save three-channel optical flow images. We provide a script to generate optical flow images using the TV-L1 algorithm:
    ```bash
    python create_optical_flow_all.py
    ```

3. AU Intensity Values

    We provide pre-computed AU intensity values (24 AUs per frame) obtained from a model pre-trained on FEAFA:
    `ME_inference_au.json`

## Training

Run the main training script:
```bash
python main_au.py --transform_au
```


Configuration Options:
| Argument | Type | Description |
| -------- | ----  | ----------- |
| --json_path | str | Path to AU intensity JSON file |
| --xlsx_path | str  | Path to micro-expression annotation file |
| --optical_flow_path | str | Path to optical flow images |
| --transform_au | flag | Whether to apply data augmentation to AUs |
| --n_frames | int  | Number of frames to use for AU features |


## Evaluation
The model performs Leave-One-Subject-Out (LOSO) cross-validation and outputs average results across all subjects. The results are printed during training.



## Acknowledgements
We acknowledge the contributions of previous works in this field, particularly the (HTnet)[https://github.com/wangzhifengharrison/HTNet]


