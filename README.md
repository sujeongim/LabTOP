# LabTOP: A Unified Model for Lab Test Outcome Prediction on Eletronic Health Records

**🏆 LabTOP was awarded the [CHIL 2025](https://chil.ahli.cc/papers/) Best Paper Award in Track 2: Applications & Practice!**

> Lab tests are fundamental for diagnosing diseases and monitoring patient conditions. However, frequent testing can be burdensome for patients, and test results may not always be immediately available. 
To address these challenges, we propose LabTOP, a unified model that predicts lab test outcomes by leveraging autoregressive generative modeling approach on EHR data.
Unlike conventional methods that estimate only a subset of lab tests or classify discrete value ranges, LabTOP performs continuous numerical predictions for a diverse range of lab items.
We evaluate LabTOP on three publicly available EHR datasets, and demonstrate that it outperforms existing methods, including traditional machine learning models and state-of-the-art large language models.
We also conduct extensive ablation studies to confirm the effectiveness of our design choices.
We believe that LabTOP will serve as an accurate and generalizable framework for lab test outcome prediction, with potential applications in clinical decision support and early detection of critical conditions.




![Training_and_Inference_Overview](https://github.com/sujeongim/LabTOP/blob/main/training_inference.png)




## Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
- `git` to clone the repository.
- A terminal (e.g., Bash, PowerShell, or Command Prompt).
- Raw EHR dataset ([MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), [eICU](https://physionet.org/content/eicu-crd/2.0/), [HiRID](https://physionet.org/content/hirid/1.1.1/))
  
## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sjim/labtop.git
cd LabTOP
```

### 2. Create the Conda Environment
Create the Conda environment:
```bash
conda env create -n labtop
conda activate labtop
python3 -m pip install torch torchvision torchaudio
```
### 3. Install all dependencies
```
conda env create -f environment.yml
```

### 4. Preprocess
```
cd labtop/src
```
```
python scripts/preprocess.py data=mimiciv data_path="path_of_raw_dataset" dest_path=../data/mimiciv
```
```
python scripts/preprocess.py data=eicu data_path="path_of_raw_dataset" dest_path=../data/eicu
```
```
python scripts/preprocess.py data=hirid data_path="path_of_raw_dataset" dest_path=../data/hirid
```

### 5. Train
```
python scripts/train.py data=mimiciv data_path=../data/mimiciv
```

### 5. Evaluate
```
python scripts/evaluate.py data=mimiciv data_path=../data/mimiciv
```
---
## Citation

Feel free to cite us if you like LabTOP.

```bibtex
@article{im2025labtop,
  title={LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health Records},
  author={Im, Sujeong and Oh, Jungwoo and Choi, Edward},
  journal={arXiv preprint arXiv:2502.14259},
  year={2025}
}
