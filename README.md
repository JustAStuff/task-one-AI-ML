# task-one-AI-ML

# Titanic Data Preprocessing

This is a simple project that demonstrates how to clean and prepare raw data from the Titanic dataset taken from Kaggle for machine learning tasks using Python.

Objective
  The main objective of this project:-

To clean the Titanic dataset by:
- Handling missing values
- Encoding categorical features
- Normalizing numerical features
- Detecting and removing outliers

Tools Used
- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

Dataset
Download `titanic-dataset.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) and place it in your project folder.

Steps Performed
1. Loaded the Titanic dataset
2. Filled missing values (`Age`, `Embarked`)
3. Encoded `Sex` and `Embarked` columns
4. Scaled `Age` and `Fare` using StandardScaler
5. Visualized and removed outliers from `Fare`

How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python sample.py
