# 🏦 Loan Approval Prediction Using Machine Learning and Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-green.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0+-yellow.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.42+-lightblue.svg" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>AI-powered loan approval system with transparent decision-making using Ensemble Learning and SHAP Explainability</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Explainable AI](#-explainable-ai)
- [Screenshots](#-screenshots)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Overview

This project implements an **intelligent loan approval prediction system** that combines multiple machine learning algorithms through ensemble learning to provide accurate and transparent loan eligibility decisions. The system leverages **Explainable AI (XAI)** techniques using SHAP (SHapley Additive exPlanations) to make the decision-making process transparent and interpretable.

**Key Highlights:**
- 🔗 **Hybrid Ensemble Learning** combining Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM
- 📊 **Real-time prediction dashboard** built with Streamlit
- 🔍 **SHAP-based explainability** for transparent AI decisions
- 📈 **Comprehensive model evaluation** with accuracy, precision, recall, and F1-score metrics
- 🎨 **Interactive data visualizations** including correlation heatmaps and feature importance

**Project Type:** Final Year Project  
**Institution:** University College of Engineering Villupuram (UCEV) - Anna University, Chennai  
**Department:** Information Technology (2026)  
**Guided By:** Dr. C. Saravanakumar, Department of IT, UCEV

---

## ✨ Features

### 1. 🤖 Multi-Model Ensemble Prediction
- **5 Individual Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- **Voting Classifier:** Soft voting ensemble for robust final predictions
- **Probability-based decisions:** Shows approval probability for each model

### 2. 🔍 Explainable AI (XAI)
- **SHAP Analysis:** Feature impact visualization showing which factors influenced the decision
- **Green/Red indicators:** Positive (green) and negative (red) feature contributions
- **Transparency:** Users understand WHY a loan was approved or rejected

### 3. 📊 Interactive Dashboard
- **3 Tab Interface:**
  - 🎯 **Loan Prediction** - Input applicant details and get instant results
  - 📈 **Correlation of Features** - Feature correlation heatmap & Random Forest importance
  - 📉 **Evaluation Metrics** - Model performance comparison table and charts

### 4. 📝 Applicant Input Form
- Monthly Income & Co-applicant Income
- Loan Amount & Term
- Demographics: Gender, Marital Status, Dependents
- Education, Employment Status, Credit History
- Property Area (Rural/Semiurban/Urban)

### 5. 📈 Data Visualization
- **Correlation Heatmap** - Seaborn heatmap of feature relationships
- **Feature Importance** - Random Forest feature ranking
- **Performance Comparison** - Bar chart comparing all model metrics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Loan      │  │  Correlation    │  │   Evaluation    │  │
│  │ Prediction  │  │    Heatmap      │  │    Metrics      │  │
│  └─────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING MODULE                 │
│  • Missing Value Handling (Mode/Median imputation)           │
│  • Label Encoding (Categorical variables)                    │
│  • Standard Scaling (StandardScaler)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING MODULE                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Logistic  │ │   Decision  │ │   Random    │            │
│  │ Regression  │ │    Tree     │ │   Forest    │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐                            │
│  │   XGBoost   │ │   LightGBM  │                            │
│  └─────────────┘ └─────────────┘                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         VOTING CLASSIFIER (Ensemble)                │    │
│  │              Soft Voting Aggregation                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY MODULE                     │
│  • SHAP Kernel Explainer                                     │
│  • Feature Impact Visualization                              │
│  • Decision Transparency                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📖 Usage

### Running the Loan Prediction System

1. **Start the application** using the command above
2. **Enter applicant details** in the left sidebar:
   - Financial: Monthly Income, Co-applicant Income, Loan Amount
   - Personal: Gender, Marital Status, Dependents, Education
   - Professional: Self Employed status, Credit History
   - Property: Loan Term, Property Area
3. **Click "Predict Eligibility"** button
4. **View Results:**
   - Individual model predictions with probabilities
   - Final ensemble decision (Approved/Rejected)
   - SHAP feature impact explanation

### Exploring Analytics

- **Tab 2 - Correlation of Features:** View feature relationships and importance
- **Tab 3 - Evaluation Metrics:** Compare model performance across all metrics

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── 📄 app.py                          # Main Streamlit application
├── 📊 dataset.csv                     # Loan approval dataset (614 records)
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
├── 📄 LICENSE                         # MIT License
│
├── 📁 assets/                         # Screenshots and images
│   ├── screenshot_prediction.png
│   ├── screenshot_correlation.png
│   └── screenshot_metrics.png
│
├── 📁 docs/                           # Additional documentation
│   ├── Final_Review.pptx              # Project presentation
│   └── project_report.pdf
│
└── 📁 models/                         # Saved model files (optional)
    └── ensemble_model.pkl
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.8127 | 0.7935 | 0.9834 | 0.8783 |
| **Decision Tree** | 0.8322 | 0.8196 | 0.9692 | 0.8882 |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **LightGBM** | 0.9919 | 0.9883 | 1.0000 | 0.9941 |
| **Ensemble (Voting)** | **0.9919** | **0.9883** | **1.0000** | **0.9941** |

> **Note:** Random Forest and XGBoost achieved perfect scores on training data. The Ensemble Voting Classifier provides robust generalization by combining all models.

---

## 🔬 Explainable AI

### SHAP (SHapley Additive exPlanations)

The system uses **SHAP Kernel Explainer** to provide interpretable explanations:

- **Green bars:** Features that INCREASE approval probability (positive impact)
- **Red bars:** Features that DECREASE approval probability (negative impact)
- **Feature ranking:** Shows which factors mattered most for the specific prediction

### Key Influencing Factors
Based on feature importance analysis:
1. **Credit History** - Most critical factor (25%+ importance)
2. **Applicant Income** - Second most important (20%+ importance)
3. **Loan Amount** - Third factor (15%+ importance)
4. **Co-applicant Income** - Supporting factor
5. **Property Area** - Location-based risk assessment

---

## 📸 Screenshots

### 🎯 Loan Prediction Dashboard
![Loan Prediction](assets/screenshot_prediction.png)

### 📈 Feature Correlation & Importance
![Correlation Analysis](assets/screenshot_correlation.png)

### 📊 Model Evaluation Metrics
![Performance Metrics](assets/screenshot_metrics.png)

---

## 🛠️ Technologies Used

### Core Framework
- **Python 3.8+** - Primary programming language
- **Streamlit** - Interactive web application framework

### Machine Learning Libraries
- **scikit-learn** - Logistic Regression, Decision Tree, Random Forest, Voting Classifier, StandardScaler, LabelEncoder
- **XGBoost** - Extreme Gradient Boosting classifier
- **LightGBM** - Light Gradient Boosting Machine

### Explainability & Visualization
- **SHAP** - Model explainability and feature impact analysis
- **Matplotlib** - Static plotting and chart generation
- **Seaborn** - Statistical data visualization (heatmaps)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Data Processing
- **StandardScaler** - Feature normalization
- **LabelEncoder** - Categorical variable encoding
- **Train-Test Split** - Model validation strategy

---

## 📚 Dataset

**Source:** Loan Approval Dataset (614 records)  
**Features:** 12 input variables + 1 target variable

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `loan_id` | String | Unique identifier for each loan application |
| `gender` | Categorical | Male/Female |
| `married` | Categorical | Marital status (Yes/No) |
| `dependents` | Categorical | Number of dependents (0, 1, 2, 3+) |
| `education` | Categorical | Graduate/Not Graduate |
| `self_employed` | Categorical | Self employment status (Yes/No) |
| `applicantincome` | Numeric | Monthly income of applicant (₹) |
| `coapplicantincome` | Numeric | Monthly income of co-applicant (₹) |
| `loanamount` | Numeric | Loan amount in thousands (₹) |
| `loan_amount_term` | Numeric | Loan term in months |
| `credit_history` | Binary | Credit history meets guidelines (1=Good, 0=Poor) |
| `property_area` | Categorical | Property location (Urban/Semiurban/Rural) |
| `loan_status` | Target | Loan approval status (Y=Approved, N=Rejected) |

### Data Preprocessing Pipeline
1. **Missing Value Treatment:**
   - Categorical: Mode imputation
   - Numerical: Median imputation (LoanAmount)
2. **Encoding:** Label encoding for categorical variables
3. **Scaling:** StandardScaler for feature normalization
4. **Feature Engineering:** Loan amount scaled to thousands

---

## 👥 Contributors

This project was developed as part of the **Final Year Project** at **University College of Engineering Villupuram (UCEV)**, a constituent college of **Anna University, Chennai**.

### Team Members
- **Vijai K** (422522205036)
- **Sasidharan K** (422522205019)
- **Vigneshwar V** (422522205039)
- **Arulsamy** (422522205027)

### Project Guide
- **Dr. C. Saravanakumar**  
  Department of Information Technology, UCEV

### Institution
- **University College of Engineering Villupuram (UCEV)**
- **Anna University, Chennai**
- **Department of Information Technology - 2026**

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 UCEV IT Department - Loan Approval Prediction Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- **Dr. C. Saravanakumar** for continuous guidance and support throughout the project
- **Anna University, Chennai** for providing the academic framework
- **UCEV IT Department** for resources and infrastructure
- **SHAP** developers for the explainability framework
- **Streamlit** team for the amazing web app framework

---

## 📞 Contact & Support

For queries, suggestions, or collaboration:

- 📧 **Email:** [sasitharan0539@gmail.com]
- 🔗 **LinkedIn:** [ https://www.linkedin.com/in/sasidharan0539/]
- 🐙 **GitHub Issues:** [Create an issue](https://github.com/sasitharan0539/loan-approval-prediction/issues)

---

## 🔮 Future Enhancements

- [ ] Integration with real-time banking APIs
- [ ] LIME explainability alongside SHAP
- [ ] Docker containerization for deployment
- [ ] REST API using FastAPI
- [ ] Model retraining pipeline with new data
- [ ] Bias detection and fairness metrics
- [ ] Multi-language support for regional users
- [ ] Cloud deployment (AWS/GCP/Azure)

---

<p align="center">
  <b>⭐ Star this repository if you found it helpful!</b><br>
  <b>🍴 Fork it to build your own version!</b>
</p>

<p align="center">
  Made with ❤️ by UCEV IT Department Team
</p>
