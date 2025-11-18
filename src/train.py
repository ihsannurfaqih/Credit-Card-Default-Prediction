import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest

# model imports
from sklearn.linear_model import LogisticRegression #linear model
from sklearn.svm import SVC # support vector machine
from sklearn.ensemble import RandomForestClassifier #ensemble model

from sklearn.inspection import permutation_importance

from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pd.options.display.max_columns = 100



df =  pd.read_csv('../data/default_credit_dataset.csv', sep=';', dtype=str)
# renaming column
df.rename(columns={'default payment next month': 'default'}, inplace=True)

for col in [col for col in df.columns if ('PAY_' in col) or (col.startswith('AGE')) or ('BILL' in col) or ('LIMIT' in col) or ('default' in col)]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


selected_feature = [
    "PAY_AMT6"
    ,"PAY_AMT5"
    ,"PAY_AMT4"
    ,"PAY_AMT3"
    ,"PAY_AMT2"
    ,"PAY_AMT1"
    ,"BILL_AMT6"
    ,"BILL_AMT5"
    ,"BILL_AMT4"
    ,"BILL_AMT3"
    ,"BILL_AMT2"
    ,"BILL_AMT1"
    ,"PAY_6"
    ,"PAY_5"
    ,"PAY_4"
    ,"PAY_3"
    ,"PAY_2"
    ,"PAY_0"
]


X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['default']), df['default'], test_size=0.2, random_state=42, stratify=df['default'])

X_train = X_train[selected_feature].copy()
X_test = X_test[selected_feature].copy()


preprocessing_lr = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), selected_feature)
    ]
, remainder='drop'
)

final_model = Pipeline(steps=[
    ('pre', preprocessing_lr),
    ('clf', LogisticRegression(
    penalty='l1',
    C=0.1,
    max_iter=300,
    solver='liblinear',
    random_state=42,
    class_weight='balanced'
))
])

print("Training model started...")
final_model.fit(X_train, y_train)
print("Training model completed.")


print("Exporting The Model")
with open("../model/final_model.bin", "wb") as f:
    pickle.dump(final_model, f)
print("Model Exported")