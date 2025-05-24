
# 📊 A PRACTICAL STUDY OF GENERALIZED LLMs AND DOMAIN-SPECIFIC MODELS FOR FINANCIAL SENTIMENT ANALYSIS  
**Author:** Onur Tuncay

This repository contains a research project aimed at evaluating the performance and optimization potential of large language models (LLMs) in the domain of financial sentiment analysis. Specifically, the study compares generalized models (Qwen 2.5) with domain-specific models (FinBERT and FiLM), highlighting their adaptability, accuracy, and limitations in handling complex financial language.

---

## 📌 Project Overview

The project investigates how generalized and domain-specific LLMs interpret financial sentiment. It systematically applies preprocessing and fine-tuning techniques to improve classification accuracy across three sentiment categories: **positive**, **neutral**, and **negative**. The key objectives include:

- Analyzing model effectiveness using macro-averaged metrics.
- Enhancing LLM performance through preprocessing and fine-tuning.
- Comparing results with state-of-the-art financial NLP models.
- Evaluating the practical use of LLMs in real-world finance scenarios.

---

## 📂 Dataset Information

- **Name:** Financial PhraseBank  
- **Source:** [Hugging Face – Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)  
- **Format:** CSV  
- **Description:** Contains financial news snippets labeled with expert-assigned sentiment classes (positive, neutral, negative).  
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)

**Reference:**  
Malo, P., Sinha, A., Korhonen, P., Wallenius, J. and Takala, P. (2014)  
*"Good debt or bad debt: Detecting semantic orientations in economic texts"*,  
_Journal of the Association for Information Science and Technology_, 65(4), pp. 782–796.  
[DOI: 10.1002/asi.23062](https://doi.org/10.1002/asi.23062)

---

## 🧠 Models & Methods

This study explores the performance of the following models:

- **Qwen 2.5 (0.5B parameters)** – Fine-tuned  
- **FinBERT** – Domain-specific financial BERT  
- **FiLM (Financial Language Model)** – Specialized for financial tasks

### Preprocessing Techniques Applied:

- Special Characters & Punctuation Removal  
- Lowercasing  
- Financial Symbol Conversion  
- Whitespace Normalization  
- Stopword Removal  
- Financial Term Expansion  
- URL Removal  
- Finance-Aware Lemmatization  

---

## 📈 Evaluation Metrics

Due to class imbalance in the dataset, **Macro F1 Score** was selected as the primary metric.  
Other metrics included:

- **Accuracy**  
- **Precision**  
- **Recall**

---

## 💻 Experimental Setup

The experiments were conducted on Google Colab Pro+, leveraging the following hardware:

- **GPU:** NVIDIA Tesla V100 (40GB VRAM)  
- **RAM:** 83.5 GB  
- **Disk:** 235.7 GB

This environment supported:

- Efficient memory usage via gradient checkpointing  
- Mixed-precision (FP16) training  
- Large-scale transformer model fine-tuning  
- Parameter-efficient fine-tuning (LoRA)  
- Stable batch processing for long sequence input

---

## 🚀 Results & Model Evaluation

**Coming Soon:**  
Detailed evaluation results, performance metrics, and model comparison insights will be added here shortly.

---

## 🛠️ Dependencies

All dependencies are listed in the [`requirements.txt`](./requirements.txt) file.  
Key libraries include:

- `transformers`  
- `datasets`  
- `peft`  
- `torch`  
- `scikit-learn`  
- `accelerate`  
- `bitsandbytes`  
- `pandas`, `numpy`, `matplotlib`, `seaborn`

To install the dependencies:

```bash
pip install -r requirements.txt
```
## 📜 License
This project is licensed under the MIT License.
All external models and datasets used adhere to their respective open-source and research-use licenses.

## 👤 Contributor
Developed and maintained by:
Onur Tuncay – Researcher in Machine Learning & NLP,  Senior Data Scientist and MSc Data Science Student at University of Gloucestershire


## 🧠 Citation
If you use this project or its methodology in your research, please cite appropriately based on the referenced paper (will be added shortly) and GitHub repository.

