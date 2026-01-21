# DataLens  
**From raw data to actionable knowledge**

DataLens is an interactive **Exploratory Data Analysis (EDA) engine** that combines  
**deterministic data analytics** with **optional AI-powered explanations**.

Unlike typical AI-based analytics tools, DataLens **never relies on LLMs for computation**.  
All insights are derived using **transparent, rule-based logic**.  
AI is used **only to explain results and never to invent them**.

---

## Key Features

### Deterministic Data Analysis
- Dataset shape & schema inspection  
- Missing value analysis  
- Duplicate removal  
- Outlier handling using IQR  
- Summary statistics  
- Correlation detection  
- Skewness analysis  
- High-cardinality detection  

### Rule-Based Insights (No Hallucinations)
- Columns with high missing values  
- Skewed numeric features  
- Strong feature correlations  
- Dataset health score (0–100)  
- ML readiness risk assessment  

### Optional AI Explanations
- Natural-language explanations of **pre-computed insights**  
- Guardrails prevent hallucinated insights  
- AI never sees raw data ,checks only summaries  
- Can be fully disabled  

### Feature Engineering
- One-hot encoding  
- Ordinal encoding  
- Automatic datetime feature extraction  

### Reporting & Export
- Download cleaned dataset as CSV  
- Generate insight reports (TXT / PDF)  

---

## App Preview
*Screenshots from the DataLens interface*

-Dataset Overview : https://drive.google.com/file/d/1kKcPwkRjlOlGbVKnekrmO9Mr-K4rfYT0/view?usp=drive_link

-Data Health & Insights : https://drive.google.com/file/d/1ul9j4XRqiXhwZnNk0DNWq82Xiwtu-scg/view?usp=drive_link

-AI Explanation Pane: https://drive.google.com/file/d/1raagPMDmaRvGPbA0byrta48Sg6vJLgvc/view?usp=sharing

---
## App Demo Video
A short walkthrough demonstrating DataLens features, interaction flow, and insight generation.

Watch the demo:https://drive.google.com/file/d/1xVommyO2dsmNymHW8pjY_uE4NBTPscVl/view?usp=sharing

---

## Design Philosophy

> **Compute first. Explain later. Never guess.**

DataLens follows a strict separation of concerns:

| Layer | Responsibility |
|------|---------------|
| Analytics Engine | Cleaning, metrics, insights (deterministic) |
| Insight Engine | Health score, ML readiness, risk detection |
| Explanation Layer | Optional AI explanation only |
| UI Layer | Interactive exploration |

This design makes the system **auditable, trustworthy, and production-friendly**.

---
## Presentation (Architecture & Concept)
This presentation explains the system design, analytics workflow, and guardrails used to prevent hallucinations.

Download the presentation: https://drive.google.com/file/d/1eR6F1ePEhZYBzcKxpAWwl8GsjLnIeiQK/view?usp=sharing

---

## Example Use Cases
- Quick dataset health assessment  
- Pre-ML data validation  
- Business data exploration  
- Teaching EDA concepts  
- Analyst productivity tool  

---

## Tech Stack
- **Python**  
- **Streamlit** (UI)  
- **Pandas / NumPy** (Data processing)  
- **Matplotlib** (Visualizations)  
- **Scikit-learn** (Encoding)  
- **Gemini API** (Optional explanations)  
- **ReportLab** (PDF reports)  

---


## How to Run

git clone : https://github.com/NandanaVShamjith/DataLens-Deterministic-EDA

cd datalens

pip install -r requirements.txt

streamlit run app.py

---

## Author
Nandana V Shamjith
Data Analytics / Data Science Project











