# CheMeleon - OpenADMET x ExpansionRx Challenge

This repository contains a machine learning pipeline for the [OpenADMET x ExpansionRx](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) challenge, focusing on predicting various ADMET properties of chemical compounds.

## Environment Setup

### Option 1: Automated Setup (Recommended)

We provide a setup script that creates a Python environment and installs all required dependencies:

1. Make sure the script is executable:
   ```bash
   chmod +x setup_env.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_env.sh
   ```

3. Activate the environment:
   ```bash
   source chemprop_env/bin/activate
   ```

### Option 2: Manual Setup

If you prefer to set up the environment manually:

1. Install UV (a faster Python package installer):
   ```bash
   pip install uv
   ```

2. Create a Python 3.11 environment:
   ```bash
   uv venv --python=3.11 chemprop_env
   ```

3. Activate the environment:
   ```bash
   source chemprop_env/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```


## HuggingFace Authentication

Before running the training or inference scripts, you need to log in to HuggingFace:

```bash
huggingface-cli login
```

This is required to access the dataset files. After the first run, data files are cached to disk.




