# 🏠 House Price Predictor

A complete end-to-end machine learning project that predicts residential house prices using regression, ensemble, and boosting models — with an interactive prediction function and visual diagnostics built in Python.

---


**Kiran Kumar Pradhan**
Roll No: 125ID0012

---

## 📌 Project Overview

This project builds a housing price prediction pipeline on a real-world dataset of 545 Indian residential properties. It systematically explores data distributions, applies transformations, trains and tunes multiple ML models, compares them fairly, and concludes with a ready-to-use prediction function and residual diagnostics.

**Dataset:** `Housing.csv` — 545 rows × 13 columns
**Target Variable:** `price` (house price in Rs.) — predicted in log scale, converted back for output
**Problem Type:** Supervised Regression

---

## 📊 Model Results — Actual Outputs

All results are from actual notebook runs, not estimates.

| Model | Train R² | Test R² | CV R² (5-fold) | RMSE (Rs.) |
|---|---|---|---|---|
| Linear Regression | 0.7019 | 0.6594 | — | 13,12,080 |
| Ridge Regression (α=0.01) | 0.7019 | 0.6594 | — | — |
| Decision Tree (tuned) | — | — | 0.5818 | — |
| XGBoost (default) | 0.9969 | 0.6163 | — | — |
| **XGBoost (tuned)** | **0.8371** | **0.6255** | **0.6689** | **13,75,785** |
| Random Forest (GridSearchCV) | — | — | **0.6692** | — |
| Random Forest (constrained, OOB) | — | OOB: 0.6564 | — | — |
| Voting Ensemble (Ridge+GB+RF) | — | 0.6371 | 0.6816 | — |
| Gradient Boosting | — | — | — | — |
| XGBoost + PCA | 0.8987 | 0.6217 | — | — |

**✅ Final Model: Random Forest** (GridSearchCV tuned)
Best CV R² = 0.6692, highest generalisation with no extreme overfitting.

---

## 🔧 Tech Stack

```
Python 3.x         pandas, numpy
scikit-learn       XGBoost
matplotlib         seaborn


```

---

## 📁 Project Structure

```
house-price-predictor/
│
├── Housing.csv                      # Raw dataset (545 rows × 13 cols)
├── House_Price_Predictor_Task_2.ipynb   # Main notebook
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn ipywidgets joblib
```

**3. Open in Jupyter or Google Colab**
```bash
jupyter notebook House_Price_Predictor_Task_2.ipynb
```

**4. Update the dataset path in Cell 1**
```python
df = pd.read_csv('Housing.csv')   # change path if needed
```

**5. Run All Cells** — prediction function and plots will appear at the end.

---

## 🧪 Complete Notebook Walkthrough

### Step 1 — Data Loading & Exploration (Cells 1–5)
- Load `Housing.csv` using pandas
- `.sample(5)` for quick visual check
- `.shape` → 545 rows, 13 columns
- `.isnull().sum()` → zero missing values in all columns
- `.duplicated().sum()` → zero duplicate rows

### Step 2 — EDA: Distributions & Skewness (Cells 6–8)
- KDE plots for all 6 numerical features with mean line overlay
- Skewness measured for each column:

| Column | Skewness | Type |
|---|---|---|
| price | 1.21 | Highly right skewed |
| area | 1.32 | Highly right skewed |
| bathrooms | 1.59 | Highly skewed |
| stories | 1.08 | Moderately skewed |
| parking | 0.84 | Moderately skewed |
| bedrooms | 0.50 | Fairly symmetric |

### Step 3 — Feature Engineering & Log Transforms (Cells 9–11)
- Applied `np.log1p()` to: `price`, `area`, `bathrooms`
- Result: price skew reduced from **1.21 → 0.14**, area from **1.32 → 0.13**
- Compared before/after skewness side by side

### Step 4 — Correlation Analysis (Cells 13–14)
- Heatmap of numerical correlations
- Top correlations with price:

| Feature | Correlation with price |
|---|---|
| area_log | 0.561 |
| bathroom_log | 0.517 |
| stories | 0.421 |
| parking | 0.384 |
| bedrooms | 0.366 |

### Step 5 — Preprocessing (Cells 18–21)
- 80/20 train-test split (`random_state=42`)
- `StandardScaler` fitted on train only → applied to test (no leakage)
- `OneHotEncoder(drop='first')` for 7 categorical columns
- Final feature matrix: **436 × 13** train, **109 × 13** test

### Step 6 — Model Training & Evaluation

**Linear Regression** (Cells 23–27)
- Train R²: 0.7019 | Test R²: 0.6594 | Gap: 0.0425

**Ridge Regression** (Cell 28)
- α = 0.01 | Test R²: 0.6594 | Stable, no improvement over LR

**Decision Tree** (Cells 29–30)
- RandomizedSearchCV + GridSearchCV both tried
- Best CV R²: 0.5818 — worst performing model

**Random Forest** (Cell 31)
- GridSearchCV over 4×6×4×4×2 = 768 combinations
- Best: `max_depth=None, max_features='sqrt', n_estimators=100`
- Best CV R²: **0.6692**

**Constrained Random Forest** (Cell 33)
- `max_depth=8, min_samples_leaf=4` to reduce overfitting
- OOB R²: 0.6564

**XGBoost — Default** (Cell 34)
- Train R²: 0.9969 | Test R²: 0.6163 | Gap: **0.3806** — severe overfit

**XGBoost — Tuned** (Cell 35)
- RandomizedSearchCV, 100 iterations
- Best: `learning_rate=0.05, max_depth=4, subsample=0.8`
- CV R²: 0.6689 | Test R²: 0.6255 | Gap reduced to 0.2116

**Gradient Boosting** (Cell 41)
- `n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8`
- Used as component in ensemble

**Voting Ensemble** (Cell 32)
- Ridge + GradientBoosting + RandomForest averaged
- Test R²: 0.6371 | CV R²: 0.6816 ± 0.0244
- High CV but lower test → overfitting signal on 545-row dataset

### Step 7 — PCA Analysis (Cells 39–42)
- Tried keeping 95% variance → reduced 13 → 10 features
- Result: Test R² dropped 0.6255 → 0.6217, gap worsened
- **Verdict: PCA not needed** — all 13 features carry real signal

### Step 8 — Feature Importance (Cell 38)
- Horizontal bar chart of XGBoost feature importances
- Top contributors: `area_log`, `bathroom_log`, `prefarea`, `airconditioning`

### Step 9 — Prediction Function (Cell 43)
```python
predict_price(
    area=5750, bedrooms=3, bathrooms=2, stories=4,
    mainroad='yes', guestroom='yes', basement='no',
    hotwaterheating='no', airconditioning='yes',
    parking=1, prefarea='yes', furnishingstatus='unfurnished'
)
# Output → Predicted Price: Rs.7,204,170
```

### Step 10 — Residual Diagnostics (Cell 44)
- Actual vs Predicted scatter plot with perfect-fit diagonal
- Residual plot (predicted vs error) — checks for patterns
- Both plots in one figure for clean presentation

---

## 📈 Key Findings

- Log transformation dramatically improved normality of price and area distributions
- **Area** and **bathrooms** are the two strongest predictors
- **XGBoost without tuning massively overfit** (train 0.997, test 0.616)
- **Complex ensembles** got high CV scores but lower test R² — dataset is too small (545 rows)
- **PCA hurt performance** — 13 features are all meaningful, dimensionality reduction unnecessary
- **Random Forest** gave the best balance of CV score and generalisation

---

## ⚠️ Limitations

- 545-row dataset caps model ceiling at approximately R² ≈ 0.70
- No geographic/location features — a major real-world price driver
- All properties from a single region — limited generalisability
- Binary yes/no amenity features are low-resolution signals

