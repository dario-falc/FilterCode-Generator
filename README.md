# Filter Code Generator

Filter Code Generator is a Python-based experimental project that explores the use of Large Language Models (LLMs) to automatically generate **JavaScript filter code for IFTTT applets** starting from natural language descriptions.

The project evaluates multiple open-source code-oriented LLMs and compares their outputs using standard NLP and code-specific evaluation metrics.

---

## 📌 Project Overview

Given:
- a textual description of an IFTTT applet,
- a high-level user intent,
- available trigger variables and action methods,

the system generates **valid JavaScript filter code** that:
- allows the action to run when conditions are met,
- otherwise blocks execution using the `.skip("reason")` method.

The generated code is then evaluated against reference implementations using automatic metrics.

---

## 🧠 Models Used

The project compares multiple open-source LLMs specialized in code generation:

- **google/gemma-2-2b-it**
- **deepseek-ai/deepseek-coder-1.3b-instruct**
- **Qwen/Qwen2.5-Coder-1.5B-Instruct**

All models are loaded via the Hugging Face `transformers` library.

---

## 📂 Project Structure

```text
FilterCode-Generator/
│
├── data/
│   ├── data.json                     # Input applet descriptions and generated outputs
│   ├── new_data.json                 # Alternative / extended dataset
│   └── generated_filtercode_from_intent.jsonl
│
├── model1.py                         # Filter code generation using DeepSeek-Coder
├── model2.py                         # Filter code generation using Qwen-Coder
├── model3.py                         # Filter code generation using Gemma
│
├── prompting.py                      # Prompt-based generation using OpenAI-style API
├── data_extraction.py                # Dataset preparation utilities
├── evaluation.py                     # Automatic evaluation and metrics computation
│
├── metrics.json                      # Aggregated evaluation results
├── requirements.txt                  # Project dependencies
└── README.md

## ⚙️ Requirements

Python 3.9+
PyTorch
Hugging Face Transformers
NLP evaluation libraries (BLEU, ROUGE, METEOR, CodeBLEU)

Install dependencies with:
pip install -r requirements.txt


## ▶️ How to Run

Each model can be executed independently.

Example:

python model1.py


The script will:

Load the input applets from data/data.json

Generate JavaScript filter code using the selected model

Store the generated output back into the dataset

Repeat for other models:

python model2.py
python model3.py


## 📊 Evaluation

Automatic evaluation is performed using:

BLEU

ROUGE-1 / ROUGE-2 / ROUGE-L

METEOR

CodeBLEU

CodeBERTScore

Evaluation results are stored in:

metrics.json

🧪 Notes

The project is intended for experimental and research purposes

GPU acceleration (CUDA) is recommended but not mandatory

All code generation follows strict prompt constraints to ensure valid IFTTT filter syntax