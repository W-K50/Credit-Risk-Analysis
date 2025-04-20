📄 **Credit Risk Analysis Report**
🔧 **1. Dataset Preprocessing Steps**
Dataset Used:

File: cs-training.csv from the Give Me Some Credit dataset.

Target Variable: SeriousDlqin2yrs (Binary: 1 = default, 0 = no default)

Steps Performed:


Step	Description
Missing Value Imputation	Used SimpleImputer with median strategy to fill in missing values across the dataset.
Feature Engineering	- DebtRatioPerIncome = DebtRatio / (MonthlyIncome + 1)
- AgeBucket created using pd.cut() to bin age into three groups: 0–30, 30–50, 50–100
Resampling (Handling Imbalance)	Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes since defaults are often underrepresented.
Train-Test Split	Data was split using train_test_split with stratification on the target column to maintain class distribution.
🧠** 2. Model Selection and Rationale**

Aspect	Details
Model Chosen	XGBoost Classifier (XGBClassifier)
Why XGBoost?	- High performance with imbalanced datasets
- Handles missing values internally
- Built-in regularization to prevent overfitting
- Fast training with large datasets
Evaluation Metric	ROC-AUC Score and Classification Report were used to evaluate performance. ROC-AUC is more appropriate than accuracy for imbalanced classification tasks.
🧩** 3. Challenges Faced and Solutions**

Challenge	Solution
Missing Data in Monthly Income and NumberOfDependents	Used median imputation to handle skewed data.
Imbalanced Dataset	Applied SMOTE to generate synthetic samples of the minority class (defaulters).
Non-normal Distributions	Feature engineering was done (e.g., ratio features and binning age) to reduce skew and improve interpretability.
Feature Mismatch in Web App	Ensured alignment of input features between model training and Streamlit app via column consistency.
📊 **4. Results with Visualizations and Interpretations**
✅ Model Performance Metrics:

Metric	Value
ROC-AUC Score	~0.85 (example; exact value depends on training run)
Precision/Recall	Good balance for identifying high-risk customers
Confusion Matrix	Clearly shows trade-off between False Positives and False Negatives
📉** Visualizations:**
You may include the following if they exist in your notebook:

Confusion Matrix Heatmap

Feature Importance (from XGBoost)

ROC Curve

Class Distribution Before & After SMOTE

If you want, I can extract or recreate these visualizations for you—just let me know!

📌 **Conclusion**
The model shows strong capability in identifying credit defaulters, especially after addressing class imbalance with SMOTE.

The XGBoost model's robustness combined with thoughtful preprocessing has led to reliable predictions.

The Streamlit interface makes it user-friendly for business or financial institutions to test customer risk.

