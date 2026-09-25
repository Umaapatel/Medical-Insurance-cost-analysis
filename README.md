# Regression analysis to predict medical insurance charges based on policyholder attributes

Exploratory data analysis and regression modeling to identify the key drivers of medical insurance charges and predict costs based on policyholder attributes.

## Dataset

This project uses a filtered and modified version of the Medical Insurance Price Prediction dataset from Kaggle, available under the CC0 1.0 Universal License.

| Column | Description |
|---|---|
| age | Age of the insured (integer) |
| gender | 1 = Female, 2 = Male |
| bmi | Body Mass Index (float) |
| no_of_children | Number of children (integer) |
| smoker | 0 = Non-smoker, 1 = Smoker |
| region | 1 = Northwest, 2 = Northeast, 3 = Southwest, 4 = Southeast |
| charges | Insurance charges in USD (float) |

Note: the source documentation lists smoker as 1 = Smoker / 2 = Non-smoker, but the actual data uses 0 = Non-smoker / 1 = Smoker. The analysis follows the values observed in the data.

## Data Cleaning

- Loaded with no header row and "?" used as the missing-value marker
- Removed 1,424 duplicate rows out of 2,772 total records
- Imputed 4 missing age values using the column mean
- Imputed 7 missing smoker values using the mode
- Rounded charges to 2 decimal places

## Exploratory Analysis

- Smoker status is the strongest driver of insurance charges by a wide margin
- age and bmi show moderate positive relationships with charges
- gender and region show weak relationships with charges

## Modeling Approach

Models were evaluated on a held-out 20% test split to report honest, generalizable performance.

| Model | Features | Test R2 |
|---|---|---|
| Simple Linear Regression | Smoker only | 0.555 |
| Multiple Linear Regression | All attributes | 0.676 |
| Ridge Regression (alpha=1, via 5-fold CV) | All attributes | 0.676 |
| Ridge + Polynomial (degree=2, alpha=1) | All attributes | 0.784 |

Alpha was selected using RidgeCV with 5-fold cross-validation over a range of 0.001-1000, which identified alpha=1 as optimal. Performance is stable for alpha <= 1, with degradation only at alpha >= 10.

## Results

The final model achieves a test R2 of 0.784, explaining roughly 78% of the variance in insurance charges on unseen data.

Key takeaway: Smoking status is the single strongest predictor of insurance charges, but combining all attributes with a nonlinear (polynomial) transformation captures additional predictive signal.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (LinearRegression, Ridge, RidgeCV, PolynomialFeatures, Pipeline)
- Matplotlib, Seaborn

## Author

Uma Patel
