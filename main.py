
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Machine Learning Models
import lightgbm as lgb

# Hyperparameter Tuning
import optuna

# General settings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)

print("Libraries imported successfully!")

print("\nLoading data...")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    print("Error: Make sure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the same directory.")
    exit()

# For easier preprocessing, combine train and test sets
# We will separate them again before modeling
train_ids = train_df['id']
test_ids = test_df['id']
train_target = train_df['y']

# Drop unnecessary columns
train_df = train_df.drop(['id', 'y'], axis=1)
test_df = test_df.drop('id', axis=1)

print("Data loaded. Train shape:", train_df.shape, "Test shape:", test_df.shape)

print("\n--- Basic Data Info ---")
print(train_df.info())


print("\nStarting Exploratory Data Analysis (EDA)...")

# Target variable distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=train_target)
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Subscribed to Term Deposit (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Identify numerical and categorical features
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_features = train_df.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Distributions of numerical features
print("\nPlotting distributions of numerical features...")
for col in numerical_features:
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=train_df[col])
    plt.title(f'Box Plot of {col}')
    
    plt.tight_layout()
    plt.show()

# Distributions of categorical features
print("\nPlotting distributions of categorical features...")
for col in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=train_df[col], order=train_df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
corr_matrix = train_df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# =============================================================================
print("\nStarting Feature Engineering...")

# Combine for easier feature creation
full_df = pd.concat([train_df, test_df], axis=0)

# --- Create new features ---

# Interaction Features
# Add a small epsilon to avoid division by zero
epsilon = 1e-6
full_df['balance_per_age'] = full_df['balance'] / (full_df['age'] + epsilon)
full_df['duration_per_campaign'] = full_df['duration'] / (full_df['campaign'] + epsilon)

# Binning 'age' into categories
full_df['age_group'] = pd.cut(full_df['age'], 
                              bins=[17, 30, 45, 60, 100], 
                              labels=['Young', 'Middle-Aged', 'Senior', 'Elderly'])

# --- Handle Categorical Features ---
# Using One-Hot Encoding for simplicity and robustness
# This will convert all object and category type columns
print("Applying One-Hot Encoding...")
full_df = pd.get_dummies(full_df, drop_first=True)

# Separate back into train and test sets
X = full_df[:len(train_df)]
X_test = full_df[len(train_df):]
y = train_target

print("Feature Engineering complete. New shape of X:", X.shape)

# =============================================================================
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# =============================================================================
print("\nStarting Hyperparameter Tuning with Optuna... (This may take a few minutes)")

def objective(trial):
    """Define the objective function for Optuna."""
    
    # Define the search space for hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        scores.append(auc)
        
    return np.mean(scores)

# Create a study object and optimize the objective function.
optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress verbose logs
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Increase n_trials for better results (e.g., 100-200)

print("\nHyperparameter tuning finished.")
print("Best trial score (AUC):", study.best_value)
print("Best trial parameters:", study.best_params)
best_params = study.best_params

# =============================================================================
# 6. FINAL MODEL TRAINING AND PREDICTION
# =============================================================================
print("\nTraining final model with best parameters...")

# Add fixed parameters
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['verbosity'] = -1
best_params['boosting_type'] = 'gbdt'
best_params['random_state'] = 42
best_params['n_jobs'] = -1

# Initialize and train the final model on the ENTIRE training data
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)

print("Final model trained.")

# Make predictions on the test set
print("Making predictions on the test data...")
test_predictions = final_model.predict_proba(X_test)[:, 1]


# =============================================================================
# 7. CREATE SUBMISSION FILE
# =============================================================================
print("\nCreating submission file...")

submission_df = pd.DataFrame({'id': test_ids, 'y': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Top 5 rows of submission file:")
print(submission_df.head())
