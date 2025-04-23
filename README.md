# LabTOP: A Unified Model for Lab Test Outcome Prediction on Eletronic Health Records

> Lab tests are fundamental for diagnosing diseases and monitoring patient conditions. However, frequent testing can be burdensome for patients, and test results may not always be immediately available. 
To address these challenges, we propose LabTOP, a unified model that predicts lab test outcomes by leveraging autoregressive generative modeling approach on EHR data.
Unlike conventional methods that estimate only a subset of lab tests or classify discrete value ranges, LabTOP performs continuous numerical predictions for a diverse range of lab items.
We evaluate LabTOP on three publicly available EHR datasets, and demonstrate that it outperforms existing methods, including traditional machine learning models and state-of-the-art large language models.
We also conduct extensive ablation studies to confirm the effectiveness of our design choices.
We believe that LabTOP will serve as an accurate and generalizable framework for lab test outcome prediction, with potential applications in clinical decision support and early detection of critical conditions.

![Training_and_Inference_Overview](https://github.com/sujeongim/LabTOP/blob/main/training_inference.png)



## Reproducing Guide
### Setting Environment
- Conda Env
```
conda create -n labtop python=3.10
```
```
conda activate labtop
```
- Install Requirements
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
```
pip install -r requirements.txt
```
***
### LabTOP
