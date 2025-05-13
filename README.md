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

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install required Python packages:**
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
    ```

3. pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

### Running the Application

1.  **Ensure you have pre-trained models:**
    * You can use the `ml_models.py` script to train and save example models (KNN, Random Forest, SVM) using the provided `train_data.txt`. Running this script will generate `.pkl` files (e.g., `knn_parkinson.pkl`, `rf_parkinson.pkl`, `SVM_parkinson.pkl`).
    ```bash
    python ml_models.py
    ```
    * Alternatively, use your own pre-trained models saved in `.pkl` or `.joblib` format.

2.  **Launch the Streamlit dashboard:**
    ```bash
    streamlit run frontend.py
    ```
    This will open the dashboard in your default web browser.

### 🛠️ How to Use
1.  **Upload Your Models:**
    * On the sidebar, use the file uploaders to select your trained model files (`.pkl` or `.joblib`).
    * Optionally, provide a custom name for each model.

2.  **Upload Your Dataset:**
    * In the main section, upload your test dataset (CSV or TXT format).
    * The application assumes the last column of your dataset is the target variable.

3.  **Configure Parsing Options:**
    * Select the appropriate delimiter for your dataset (e.g., comma, semicolon, tab).
    * Indicate if your dataset has a header row.
    * Specify if the first column should be used as the index.

4.  **Run Comparison Analysis:**
    * Once models and data are uploaded, click the "Run Comparison Analysis" button.

5.  **Explore Results:**
    * **Performance Metrics Tab:** View key metrics like accuracy for each model. The best model is highlighted, and a bar chart provides a visual comparison.
    * **Confusion Matrices Tab:** Examine the confusion matrix for each model. You can download the plot for each matrix.
    * **Detailed Comparison Tab:**
        * See a sample-by-sample comparison of true values versus predictions from each model.
        * If "Show Detailed Metrics" is enabled in the sidebar and multiple models are loaded, a model agreement heatmap is displayed, showing the percentage of predictions on which pairs of models agree.
        * Statistics on overall model agreement and correct agreements are also provided.

### 📁 Code Breakdown
### `frontend.py`

-   Handles the Streamlit user interface, file uploads, and display of results.
-   Includes helper functions for:
    -   `preprocess_data()`: Basic preprocessing for the input dataset (currently samples 100 rows and scales features; **customize this for your specific data**).
    -   `get_model_predictions()`: Loads uploaded models and generates predictions.
    -   `plot_confusion_matrix()`: Creates confusion matrix plots.
    -   `calculate_metrics()`: Computes performance metrics like accuracy.
-   Uses custom CSS for enhanced visual styling.

### `ml_models.py`

-   A script to demonstrate training and saving three types of machine learning models:
    -   K-Nearest Neighbors (KNN)
    -   Random Forest
    -   Support Vector Machine (SVM)
-   Uses `train_data.txt` for training.
-   Performs basic hyperparameter tuning for KNN and Random Forest.
-   Saves the trained models using `joblib` into `.pkl` files.
-   Generates and saves a correlation matrix heatmap (`Correlation Matrix Test Data.png`).

### Data

-   `train_data.txt`: A plain text file where each row represents a sample and columns are feature values, with the last column being the target variable. The delimiter appears to be a comma.
-   `Correlation Matrix Test Data.jpg`: An image showing a heatmap of feature correlations, likely generated during the model training/exploration phase.

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

-Yashas Raina

-Ritwik Mittal

### 📄 License
This project is open-source. Refer to the LICENSE file for usage guidelines. If none is provided, it is intended for educational and research purposes.







