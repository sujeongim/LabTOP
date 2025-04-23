# LabTOP: A Unified Model for Lab Test Outcome Prediction on Eletronic Health Records

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
  
## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sjim/labtop.git
cd your-repo
```

### 2. Create the Conda Environment
Create the Conda environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

This sets up a Conda environment named `your-project-env` with Python and all dependencies.

### 3. Activate the Environment
Activate the Conda environment:
- **Linux/macOS**:
  ```bash
  conda activate your-project-env
  ```
- **Windows**:
  ```bash
  conda activate your-project-env
  ```

### 4. Set Up Environment Variables
Copy the example environment file and fill in the required values:
```bash
cp .env.example .env
```
Edit `.env` with your text editor (e.g., `nano`, `vim`, or VS Code) and add your values:
```plaintext
API_KEY=your-api-key
DATABASE_URL=your-database-url
```

### 5. Run the Setup Script
Run the provided setup script to verify the environment:
```bash
bash setup.sh
```
For Windows, use:
```bash
setup.bat
```
