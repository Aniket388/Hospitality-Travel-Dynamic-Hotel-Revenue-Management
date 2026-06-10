# 🏨 Dynamic Hotel Revenue Management Engine (v2.0-Production)

[![Streamlit App](https://static.streamlit.io/badge-streamlit.svg)](https://PASTE_YOUR_STREAMLIT_APP_URL_HERE)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Project Title:** Dynamic Hotel Revenue Management: Optimising Room Pricing Strategies Across Seasons and Occupancy Clusters Using K-Means Demand Clustering and Decision Tree Pricing Rules

An enterprise-grade, decoupled dual-model machine learning pipeline engineered to optimize hotel Average Daily Rate (ADR). This system bypasses rigid, human-derived heuristic pricing models by leveraging unsupervised behavioral profiling and optimized, non-linear cross-validated regression decision nodes.

---

## 🚀 Live Interactive Dashboard
👉 **[Click Here to Access the Deployed Production Platform](https://PASTE_YOUR_STREAMLIT_APP_URL_HERE)**

---

## 🧠 System Architecture & Workflow

The platform implements a **Decoupled Dual-Model Machine Learning Pipeline** designed to bridge the gap between behavioral pattern recognition and deterministic room-rate execution.

```text
[ Raw Transaction Logs (119,390 Rows) ]
                  │
                  ▼
   [ Stage 1: Data Cleaning Engine ] ──► Outlier / Cancellation Purge
                  │
                  ▼
 [ Stage 2: Feature Engineering Core ] ──► Cyclical Unit Circle Transforms
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
[ Clustering Matrix ] [ Regression Matrix ]
       │                     │
       ▼                     │
[ Robust Scaling ]           │
       │                     │
       ▼                     │
[ Unsupervised K-Means ]     │
       │                     │
       ▼                     ▼
[ 3 Sorted Demand States ] ──► [ One-Hot Category Matrix Encoding ]
                             │
                             ▼
                    [ GridSearchCV Tuning ]
                             │
                             ▼
                    [ Decision Tree Regressor ]
                             │
                             ▼
                  [ Deployed Live App ] (Streamlit Cloud Cloud Environment)
