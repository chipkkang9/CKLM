# Setup guide: Gemma 3 4B local RAG on Apple Silicon

## 1) Create the conda environment
```bash
conda env create -f environment.yml
conda activate CKLM
```

## 2) Upgrade pip tooling
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install Python dependencies
```bash
pip install -r requirements.txt
```

## 4) Accept Gemma license and log in to Hugging Face
Before downloading Gemma, accept the model license on Hugging Face.
Then log in locally:
```bash
huggingface-cli login
```

## 5) Enable MPS fallback for Apple Silicon
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
To keep it persistent in zsh:
```bash
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc
```

## 6) Quick smoke test
Let's get STARTED!!
```bash
python quick_test.py
```
