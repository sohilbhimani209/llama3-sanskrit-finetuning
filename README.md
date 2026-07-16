# Fine-Tuning Llama-3.2-1B-Instruct for Sanskrit-to-English Translation using LoRA

This project demonstrates parameter-efficient fine-tuning (PEFT) of **Meta's Llama-3.2-1B-Instruct** model using **LoRA (Low-Rank Adaptation)** for Sanskrit-to-English machine translation.

The project was implemented using **Hugging Face Transformers**, **PEFT**, and **PyTorch**, and trained on Google Colab with an NVIDIA Tesla T4 GPU.

---

## Project Highlights

- Fine-tuned **Llama-3.2-1B-Instruct**
- Used **LoRA** for parameter-efficient fine-tuning
- Built using **Hugging Face Transformers**
- Trained on a **Tesla T4 GPU**
- Supports **Sanskrit → English Translation**
- Includes training notebook, evaluation script, and sample outputs

## Repository Structure

```text
llama3-sanskrit-finetuning/
│
├── notebook/              # Google Colab notebook
├── checkpoints/           # LoRA adapter and tokenizer files
├── evaluation/            # Evaluation script
├── sample_outputs/        # Sample translations
├── report/                # Project report
├── images/                # Images used in README
├── requirements.txt
├── README.md
└── .gitignore
```
## Problem Statement

Machine translation for low-resource languages such as Sanskrit is a challenging Natural Language Processing (NLP) task due to the limited availability of high-quality parallel datasets.

The objective of this project is to adapt a pre-trained Large Language Model (LLM) to accurately translate Sanskrit sentences into English using parameter-efficient fine-tuning. Instead of updating all model parameters, LoRA was used to significantly reduce computational cost while maintaining strong translation performance.

## Model Information

| Parameter | Value |
|-----------|-------|
| Base Model | Meta Llama-3.2-1B-Instruct |
| Fine-Tuning Method | LoRA |
| Framework | Hugging Face Transformers |
| Language Pair | Sanskrit → English |
| GPU | NVIDIA Tesla T4 |
| Precision | FP16 |

## Dataset

The Sanskrit-English translation dataset was converted into an instruction-following format suitable for supervised fine-tuning.

Example:

**User**

Translate the following Sanskrit sentence into English.

Sanskrit:

अहं विद्यालयं गच्छामि।

**Assistant**

I go to school.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Learning Rate | 2e-4 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Optimizer | AdamW |
| Max Sequence Length | 256 |
| Fine-Tuning Method | LoRA |