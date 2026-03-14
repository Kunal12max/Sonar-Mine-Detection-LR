# Sonar: Rock vs. Mine Classification ⚓
## Project Overview
A binary classification Machine Learning model built to predict whether a sonar signal is bouncing off a naval Mine (1) or a harmless Rock (0). The project focuses on handling continuous acoustic data and optimizing for **Recall** rather than just raw accuracy, keeping the real-world defense scenario in mind.
## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, NumPy
## What I Did in This Project
1. **Data Preprocessing:** Handled raw CSV data without headers to prevent data loss. Mapped categorical targets ('M' and 'R') to binary (1 and 0).
2. **Feature Scaling:** Applied `StandardScaler` to all 60 acoustic frequency columns. **Note:** Scaling was done strictly *after* the train-test split (`fit_transform` on train, `transform` on test) to prevent data leakage.
3. **Hyperparameter Tuning:** Instead of manual guessing, I implemented `RandomizedSearchCV` to find the most optimal parameters for the Logistic Regression model.
## Model Tuning Results
The Randomized Search tested multiple combinations and found the following best parameters:
* `solver`: 'liblinear' (Best engine for small datasets)
* `penalty`: 'l2' (Ridge Regularization - retained all frequency features but shrank the noisy ones)
* `C`: 0.1 (Strict regularization to prevent overfitting)
## Final Evaluation & Business Logic
* **Overall Accuracy:** 71.43%
* **Mine (Class 1) Recall:** **83%**
### Why Recall Matters Here
In a naval defense scenario, a **False Negative** (predicting a Mine is just a Rock) results in a destroyed submarine. A **False Positive** (predicting a Rock is a Mine) just means taking a cautious detour. 
Therefore, I optimized and evaluated this model based on **Recall**. The model successfully identified 83% of the actual mines in the unseen test data, proving that the `l2` penalty and `0.1` threshold created a strong safety net.
## Confusion Matrix
||Predicted:Rock(0)|Predicted: Mine (1)|
|**Actual: Rock (0)|15(True Negatives)|9 (False Positives)|
| **Actual: Mine (1)** |3(False Negatives)|15(True Positives)|
