# Grainger ML Interview

This repository contains a solution for Grainger ML Engineer take-home interview assignment.


# Evaluation for Query-Product Matching

This project explores how different embedding models and data sampling strategies affect retrieval performance in a query-to-product matching task. We benchmark models using common metrics like Hits@K and MRR to understand generalization and overfitting tendencies.

---

## Project Structure

- `data.py`  
  Generates the evaluation dataset by sampling query-product pairs. Includes:
  - Default random sampling
  - Clustered query selection
  - Stratified sampling for balanced evaluation

- `main.py`  
  Builds a FAISS index on the product embeddings and evaluates retrieval using:
  - Hits@1 / Hits@5 / Hits@10
  - Mean Reciprocal Rank (MRR)

---

## Embedding Model Comparison

We benchmarked multiple models using different data creation strategies. Below is a visual summary:

![Evaluation Table](src/main/data/output/evaluation.png)

---

## 🧪 How to Run

1. **Generate Dataset**
   ```bash
   python data.py

2. **Build Index & Evaluate** 

```bash
   python main.py
