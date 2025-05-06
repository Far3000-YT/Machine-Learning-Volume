import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #this might not be needed if we stick to chronological split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib #for saving model and scaler
import os

#print xgboost version to be sure
print(f"--- Using XGBoost version: {xgb.__version__} ---")

#load the data with features and target
input_file_processed = 'data/processed/btc_usdt_1m_features.parquet'

#check if file exists
if not os.path.exists(input_file_processed):
    #print error file not found
    print(f"Error: Processed data file not found at {input_file_processed}. Please run data_process.py first.")
    exit() #exit if no file

#read parquet into dataframe
df = pd.read_parquet(input_file_processed)

print("--- Loaded Processed Data Info ---")
#df.info() #can be verbose for many features, let's print shape and head
print(f"Shape: {df.shape}")
print("\n--- Loaded Processed Data Head ---")
print(df.head())

#CRITICAL: split chronologically, no random shuffle
#using 70% train, 15% validation, 15% test split roughly

train_size_ratio = 0.7
val_size_ratio = 0.15
#test_size_ratio is implicitly 0.15

n = len(df)
train_idx_end = int(n * train_size_ratio)
val_idx_end = train_idx_end + int(n * val_size_ratio)

train_df = df.iloc[:train_idx_end]
val_df = df.iloc[train_idx_end:val_idx_end]
test_df = df.iloc[val_idx_end:]

print(f"\nTrain data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")

#define features (X) and target (y)
#target column is 'target'
#all other columns are potential features
#explicitly remove 'future_close' if it's still present and not part of features
#and 'open', 'high', 'low', 'close', 'timestamp' if they were carried over and not intended as direct features
#based on data_process.py, only 'target' and engineered features should be there, plus original ohlcv for reference
#let's assume df from btc_usdt_1m_features.parquet contains only engineered features and 'target'
#and potentially original 'open', 'high', 'low', 'close', 'volume' for context, but they shouldn't be scaled if not features
#from your data_process.py, features are derived from ohlcv, so ohlcv itself might be redundant or should be explicitly selected/excluded
#for now, assuming all columns except 'target' and 'future_close' (which should be dropped after target creation) are features.
#data_process.py already drops future_close implicitly via dropna after shifting, or it should be explicitly dropped
#let's refine feature list based on typical practice for processed file
potential_features = [col for col in df.columns if col not in ['target', 'future_close', 'timestamp']]
#if 'open', 'high', 'low', 'close', 'volume' are direct inputs, include them.
#if they were only used to create other features, they might not be needed here.
#data_process.py seems to keep them. Let's assume they are part of the feature set for now.

#let's re-evaluate the feature list from data_process.py's output
#it seems 'open', 'high', 'low', 'close', 'volume' columns ARE kept in the processed file
#and are used as inputs to the model along with the engineered features
features = [col for col in df.columns if col not in ['target', 'future_close']] #future_close should have been handled by dropna due to shift
if 'future_close' in train_df.columns: #defensive check
    train_df = train_df.drop(columns=['future_close'])
    val_df = val_df.drop(columns=['future_close'])
    test_df = test_df.drop(columns=['future_close'])
    features = [col for col in features if col != 'future_close']


target_col_name = 'target'

X_train, y_train = train_df[features], train_df[target_col_name]
X_val, y_val = val_df[features], val_df[target_col_name]
X_test, y_test = test_df[features], test_df[target_col_name] #keep test separate for final eval

#scale features, FIT scaler only on training data
scaler = StandardScaler()

#fit on training features only
X_train_scaled = scaler.fit_transform(X_train) #fit and transform training set

#transform val and test sets using the fitted scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


#using XGBoost Classifier for binary prediction
#simple parameters for now, can tune later (Phase 3)
model = xgb.XGBClassifier(
    objective='binary:logistic', #for binary classification
    n_estimators=100, #number of trees, can be increased with early stopping
    learning_rate=0.1, #step size shrinkage
    eval_metric='logloss', #metric for evaluation during training and for early stopping
    random_state=42, #for reproducibility
    #use_label_encoder=False, #deprecated and not needed for integer labels
    #tree_method='hist' #often faster for large datasets, default is usually 'auto' which might pick it
)

print("\n--- Training Model ---")
#train the model
#use early_stopping_rounds directly in fit
#it will monitor 'logloss' on the validation set (X_val_scaled, y_val)
#XGBoost typically names it validation_0-logloss in verbose output
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)], #evaluate on validation set
    early_stopping_rounds=10, #stop if validation logloss doesn't improve for 10 rounds
    verbose=True #print training progress, set to False or a number (e.g., 10) for less output
)


print("\n--- Evaluating Model on Validation Set ---")
#make predictions on validation set
y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1] #get probabilities for the positive class
y_val_pred = model.predict(X_val_scaled) #get class predictions (0 or 1)

#calculate metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, zero_division=0) #added zero_division
recall = recall_score(y_val, y_val_pred, zero_division=0) #added zero_division
f1 = f1_score(y_val, y_val_pred, zero_division=0) #added zero_division

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}") #Precision for class 1
print(f"Validation Recall: {recall:.4f}") #Recall for class 1
print(f"Validation F1 Score: {f1:.4f}") #F1 for class 1

#more detailed report
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, zero_division=0))

#create models directory if not exists
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

#define model and scaler file paths
model_file = os.path.join(model_dir, 'xgb_price_direction_model.joblib')
scaler_file = os.path.join(model_dir, 'scaler.joblib')

#save the trained model
joblib.dump(model, model_file)
print(f"\nTrained model saved to {model_file}")

#save the scaler
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")


#feature importance
print("\n--- Feature Importances ---")
importances = model.feature_importances_
feature_names = X_train.columns #get feature names
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df.head(20)) #print top 20 features

#save feature importances to a file
feature_importance_file = os.path.join(model_dir, 'feature_importances.csv')
feature_importance_df.to_csv(feature_importance_file, index=False)
print(f"Feature importances saved to {feature_importance_file}")


print("\n--- Model Training Script Finished ---")
#next steps according to README:
#- hyperparameter tuning (Phase 3)
#- rigorous evaluation on the test set (X_test_scaled, y_test) during backtesting (Phase 4)