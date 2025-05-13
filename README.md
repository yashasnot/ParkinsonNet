# ParkinsonNet: Machine Learning Models for Parkinson's Disease Detection

A machine learning-based diagnostic tool to classify Parkinson’s Disease using biomedical voice measurements. This project demonstrates dynamic and interactive **Streamlit-based web application** designed to evaluate and compare the performance of multiple pre-trained machine learning models on a given test dataset. This tool is ideal for ML practitioners and researchers who want to analyze classification models with ease and visual clarity.

## Overview

**Built an interactive ML dashboard using Streamlit to compare classification models by uploading custom `.pkl` files and datasets, featuring performance metrics, confusion matrices, and model agreement heatmaps.**
Parkinson's Disease is a progressive neurological disorder. Early detection can help in better management of the condition. In this project, we use a dataset of biomedical voice measurements to classify whether a person has Parkinson’s Disease.

## 🔍 Key Features

- **Dynamic Model Upload:** Supports uploading up to three pre-trained models in `.pkl` or `.joblib` format.
- **Flexible Dataset Input:** Accepts CSV and TXT files; configurable parsing (delimiter, header, index column).
- **Interactive UI:** Built with Streamlit, styled using custom CSS for a clean user experience.
- **Comprehensive Evaluation:**
  - Dataset preview with descriptive statistics and target distribution.
  - Model performance comparison using metrics (accuracy) and visual bar charts.
  - Individual confusion matrix for each model with download support.
  - Detailed sample-wise prediction comparisons.
  - Model agreement heatmap to visualize prediction consensus across models.
    
## 🗂️ Project Structure

- `frontend.py` – Main Streamlit dashboard logic
- `ml_models.py` – Script to train and export demo models
- `train_data.txt` – Sample training dataset
- `Correlation Matrix Test Data.jpg` – Heatmap of feature correlation
  
![image](https://github.com/user-attachments/assets/0655b807-6a45-4050-a498-686a759d8524)

  ## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

git clone <repository-url>
cd <repository-directory>

pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
Running the App

### 🛠️ How to Use
- Upload Models:
Use the sidebar to upload up to 3 pre-trained ML models.

Optionally assign custom names to each model.

- Upload Test Dataset:

 Use the main window to upload your test dataset (.csv or .txt).

The last column is assumed to be the target variable.

- Configure Parsing Options:

Choose delimiter (comma, semicolon, tab, etc.)

Indicate presence of header and index column.

- Run Analysis:

 **Click “Run Comparison Analysis” to start.**

- Explore Results:

- Performance Metrics: Visual bar chart highlighting top-performing model.

- Confusion Matrices: For each model with download option.

- Prediction Comparison: Side-by-side result comparison with heatmap of agreement across models.

- Agreement Stats: Percent agreement and correct consensus rate.

### 📁 Code Breakdown
-frontend.py
Manages UI/UX, data upload, and output rendering.

-Contains helper functions:

preprocess_data(): Loads and scales test data.

get_model_predictions(): Generates predictions from uploaded models.

plot_confusion_matrix(): Visualizes model confusion matrices.

calculate_metrics(): Computes accuracy and other metrics.

## ml_models.py
Trains and saves 3 classifiers using train_data.txt:

**KNN**

**Random Forest**

**SVM**

Performs basic hyperparameter tuning.

Saves model files for dashboard testing.

Outputs a correlation matrix plot for data insight.

### 📊 Dataset
- Format: Plain text or CSV

- Delimiter: Comma

- Last column: Target label

### ✨ Customization Options
- **Preprocessing**: Modify preprocess_data() to match original training steps (scaling, encoding, missing value handling).

- **Metrics**: Extend calculate_metrics() to include F1-score, precision, recall, ROC AUC.

- **Model Formats**: Update model loader if using TensorFlow, PyTorch, etc.

- **Styling**: Further enhance layout via Streamlit's st.markdown and CSS.

### 👨‍💻 Authors

Yashas Raina
Ritwik Mittal

### 📄 License
This project is open-source. Refer to the LICENSE file for usage guidelines. If none is provided, it is intended for educational and research purposes.







