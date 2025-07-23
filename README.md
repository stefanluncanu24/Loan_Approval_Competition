

<h1>Loan Approval Prediction - Kaggle Playground Series Season 4 Episode 10</h1>

<p>This project is a solution to the Kaggle competition <a href="https://www.kaggle.com/competitions/playground-series-s4e10">Playground Series Season 4 Episode 10</a>. The goal is to build a machine learning model that can predict whether a loan will default (<code>loan_status = 1</code>) or not (<code>loan_status = 0</code>) based on various features related to the borrower and the loan.</p>

<img src="https://github.com/user-attachments/assets/fb425ccc-111e-423d-a283-0b8c59ab1dc4" alt="Loan Default Prediction">

<h2>File Explanations</h2>

<ul>
    <li><strong>main3.py</strong>: The main script containing all the code for data loading, preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and prediction.</li>
</ul>

<h2>Table of Contents</h2>

<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#data-loading">Data Loading</a></li>
    <li><a href="#data-description">Data Description</a></li>
    <li><a href="#exploratory-data-analysis">Exploratory Data Analysis</a>
        <ul>
            <li><a href="#target-variable-distribution">Target Variable Distribution</a></li>
            <li><a href="#numerical-feature-distributions">Numerical Feature Distributions</a></li>
            <li><a href="#categorical-feature-analysis">Categorical Feature Analysis</a></li>
        </ul>
    </li>
    <li><a href="#data-preprocessing">Data Preprocessing</a>
        <ul>
            <li><a href="#handling-missing-and-inconsistent-data">Handling Missing and Inconsistent Data</a></li>
            <li><a href="#encoding-categorical-variables">Encoding Categorical Variables</a></li>
        </ul>
    </li>
    <li><a href="#feature-engineering">Feature Engineering</a></li>
    <li><a href="#modeling">Modeling</a>
        <ul>
            <li><a href="#backward-feature-elimination">Backward Feature Elimination</a></li>
            <li><a href="#hyperparameter-tuning-with-optuna">Hyperparameter Tuning with Optuna</a></li>
        </ul>
    </li>
    <li><a href="#prediction-and-submission">Prediction and Submission</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
</ul>

<h2 id="overview">Overview</h2>

<p>In this project, we:</p>

<ul>
    <li>Load and preprocess the dataset.</li>
    <li>Perform exploratory data analysis to understand the data distribution and relationships.</li>
    <li>Engineer new features to enhance model performance.</li>
    <li>Use LightGBM with backward feature elimination for feature selection.</li>
    <li>Tune hyperparameters using Optuna.</li>
    <li>Make predictions on the test set.</li>
    <li>Prepare a submission file for Kaggle.</li>
</ul>

<h2 id="data-loading">Data Loading</h2>

<p>We start by downloading and extracting the competition data using the Kaggle API and read it into Pandas DataFrames.</p>

<pre><code>
# !pip install kaggle
    
!kaggle competitions download -c playground-series-s4e10

import zipfile
import os
import pandas as pd

zip_file_path = 'playground-series-s4e10.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('.')
os.remove(zip_file_path)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dataset = pd.concat([train, test])
</code></pre>

<h2 id="data-description">Data Description</h2>

<p>Here's a breakdown of what each column represents:</p>

<ul>
    <li><strong>id</strong>: A unique identifier for each loan application or borrower.</li>
    <li><strong>person_age</strong>: The age of the borrower.</li>
    <li><strong>person_income</strong>: The annual income of the borrower.</li>
    <li><strong>person_home_ownership</strong>: The housing status of the borrower.</li>
    <li><strong>person_emp_length</strong>: Years of employment with the current employer.</li>
    <li><strong>loan_intent</strong>: The purpose of the loan.</li>
    <li><strong>loan_grade</strong>: Classification of the loan based on risk.</li>
    <li><strong>loan_amnt</strong>: The amount of money borrowed.</li>
    <li><strong>loan_int_rate</strong>: The interest rate on the loan.</li>
    <li><strong>loan_percent_income</strong>: Ratio of the loan amount to the borrower's annual income.</li>
    <li><strong>cb_person_default_on_file</strong>: Indicates previous loan defaults.</li>
    <li><strong>cb_person_cred_hist_length</strong>: Length of the borrower's credit history in years.</li>
    <li><strong>loan_status</strong>: The target variable indicating loan repayment status.</li>
</ul>

<h2 id="exploratory-data-analysis">Exploratory Data Analysis</h2>

<h3 id="target-variable-distribution">Target Variable Distribution</h3>

<p>We visualize the distribution of the target variable to observe class imbalance.</p>

<pre><code>import matplotlib.pyplot as plt
import seaborn as sns
    
label_counts = train['loan_status'].value_counts()
    
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=["Not Defaulted", 'Defaulted'], colors=['green', 'red'], startangle=140, autopct='%1.1f%%')
plt.title('Distribution of Loan Status')
plt.show()
</code></pre>

![image](https://github.com/user-attachments/assets/5351373d-4c02-4261-b012-0e8024ecb4b5)

<h3 id="numerical-feature-distributions">Numerical Feature Distributions</h3>

<p>We plot histograms of numerical features grouped by the target variable to identify differences in distributions.</p>

<pre><code>
numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_columns.remove('loan_status')  # Remove target variable

colors = {0: 'green', 1: 'red'}

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    for status in train['loan_status'].unique():
        subset = train[train['loan_status'] == status]
        sns.histplot(subset[column], color=colors[status], label=["Not Defaulted", "Defaulted"][status], kde=True, stat="density", common_norm=False)
    plt.title(f'Distribution of {column} by Loan Status')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title='Loan Status')
    plt.show()
</code></pre>

<h3 id="categorical-feature-analysis">Categorical Feature Analysis</h3>

<p>We analyze the proportion of loan status across different categories in categorical features.</p>

<pre><code>
categorical_columns = train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_columns.remove('id')  # Remove id column if present

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    counts = train.groupby([column, 'loan_status']).size().unstack(fill_value=0)
    fraction = counts.div(counts.sum(axis=1), axis=0)
    fraction.plot(kind='bar', stacked=True, color=['green', 'red'])
    plt.title(f'Fraction of Loan Status for {column}')
    plt.ylabel('Fraction')
    plt.xlabel(column)
    plt.legend(title='Loan Status', labels=['Not Defaulted', 'Defaulted'])
    plt.show()
</code></pre>

![image](https://github.com/user-attachments/assets/fbe1fe54-f094-45fd-ba2f-a3e1e6026c95)

<h2 id="data-preprocessing">Data Preprocessing</h2>

<h3 id="handling-missing-and-inconsistent-data">Handling Missing and Inconsistent Data</h3>

<p>We remove unnecessary columns and duplicates to clean the dataset.</p>

<pre><code>
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

train.drop_duplicates(inplace=True)
</code></pre>

<p>We find instances where the borrower's employment length exceeds their age, which is inconsistent. We remove these entries.</p>

<pre><code># Identify inconsistent data where employment length exceeds age
inconsistent_entries = train[train['person_emp_length'] > train['person_age']]
train.drop(inconsistent_entries.index, inplace=True)
</code></pre>

<h3 id="encoding-categorical-variables">Encoding Categorical Variables</h3>

<p>We perform label encoding and one-hot encoding on categorical variables to prepare the data for modeling.</p>

<pre><code>from sklearn.preprocessing import LabelEncoder

def preprocess_data(df_train, df_test):
    label_enc = LabelEncoder()
    label_cols = ['person_home_ownership', 'loan_grade', 'cb_person_default_on_file']
    
    for col in label_cols:
        df_train[col] = label_enc.fit_transform(df_train[col])
        df_test[col] = label_enc.transform(df_test[col])
    
    df_train = pd.get_dummies(df_train, columns=['loan_intent'], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['loan_intent'], drop_first=True)
    
    return df_train, df_test

train_processed, test_processed = preprocess_data(train, test)
</code></pre>

<h2 id="feature-engineering">Feature Engineering</h2>

<p>We create new features to help the model capture complex relationships in the data.</p>

<pre><code>import numpy as np

def feature_engineering(df):
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate']
    df['income_per_year_emp'] = df['person_income'] / df['person_emp_length'].replace(0, np.nan)
    df['int_to_loan_ratio'] = df['loan_int_rate'] / df['loan_amnt'].replace(0, np.nan)
    df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']
    df['debt_to_credit_ratio'] = df['loan_amnt'] / df['cb_person_cred_hist_length'].replace(0, np.nan)
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN values resulting from division by zero
    df.fillna(0, inplace=True)
    return df

train_processed = feature_engineering(train_processed)
test_processed = feature_engineering(test_processed)
</code></pre>

<h2 id="modeling">Modeling</h2>

<p>We prepare the data for modeling by separating features and target variable.</p>

<pre><code>from sklearn.model_selection import train_test_split

X = train_processed.drop('loan_status', axis=1)
y = train_processed['loan_status']
</code></pre>

<h3 id="backward-feature-elimination">Backward Feature Elimination</h3>

<p>We use backward feature elimination to select the most important features, reducing the complexity of the model.</p>

<pre><code>from lightgbm import LGBMClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

lgbm = LGBMClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bfs = SFS(
    estimator=lgbm,
    k_features=(10, X.shape[1]),
    forward=False,
    floating=True,
    scoring='roc_auc',
    cv=cv_strat,
    n_jobs=-1,
    verbose=2
)

bfs = bfs.fit(X, y)
selected_features = list(bfs.k_feature_names_)

print(f"Selected Features ({len(selected_features)}): {selected_features}")
</code></pre>

<h3 id="hyperparameter-tuning-with-optuna">Hyperparameter Tuning with Optuna</h3>

<p>We use Optuna to perform hyperparameter tuning, optimizing the ROC AUC score.</p>

<pre><code>import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def objective_lgb(trial):
    lgbm_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, valid_idx in cv.split(X[selected_features], y):
        X_train_fold, X_valid_fold = X.iloc[train_idx][selected_features], X.iloc[valid_idx][selected_features]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        auc = roc_auc_score(y_valid_fold, y_pred)
        auc_scores.append(auc)
    
    return np.mean(auc_scores)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)
    
best_lgb_params = study_lgb.best_params
print("Best Hyperparameters:", best_lgb_params)

final_lgb = LGBMClassifier(
    **best_lgb_params,
    random_state=42,
    n_jobs=-1,
    objective='binary',
    metric='auc'
)
</code></pre>

<h2 id="prediction-and-submission">Prediction and Submission</h2>

<p>We train the final model on the entire training data and generate predictions on the test set, saving the results in the required submission format.</p>

<pre><code>
final_lgb.fit(X[selected_features], y)

test_processed_selected = test_processed[selected_features]

y_pred_lgb = final_lgb.predict_proba(test_processed_selected)[:, 1]

submission = pd.read_csv('sample_submission.csv')
submission['loan_status'] = y_pred_lgb
submission.to_csv('lgb_submission.csv', index=False)
</code></pre>

<h2 id="conclusion">Conclusion</h2>

<p>In this project, we built a machine learning model to predict loan defaults using LightGBM. Through data preprocessing, feature engineering, feature selection, and hyperparameter tuning, we aimed to improve the model's predictive performance.</p>

<p><strong>Next Steps:</strong></p>

<ul>
    <li>Explore more advanced feature engineering techniques.</li>
    <li>Address class imbalance using techniques like SMOTE or class weighting.</li>
</ul>

<h2 id="acknowledgments">Acknowledgments</h2>

<p>We would like to thank the Kaggle community and the organizers of the Playground Series competition for providing the dataset and the platform to practice and enhance our machine learning skills.</p>

</body>
</html>


