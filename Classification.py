#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:42:01 2025

@author: anyakondamani
"""
import random
random.seed(19884614)
#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import auc as sk_auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#%% Loading & Understanding the Data

data = pd.read_csv('musicData.csv')

nan_counts_col = data.isna().sum() 
print("NaN counts per column:\n", nan_counts_col)
total_nan_count = data.isna().values.sum()
print("\nTotal NaN count: ", total_nan_count)
num_nan_rows = data.isna().any(axis=1).sum()
print("Rows with at least one NaN:", num_nan_rows)
data = data.dropna() # dropping NaN rows

print(data['music_genre'].unique())
print(data['key'].unique())
print(data['mode'].unique())
print(data.dtypes)

data['mode']=data['mode'].replace({'Minor': 1, 'Major': 0}) # Making mode column binary.
data['is_sharp'] = data['key'].str.contains('#').astype(int) # New column, one hot encoding sharp (#).
data['key_base'] = data['key'].str.replace('#', '', regex=False) 
letter_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6} 
data['key_num'] = data['key_base'].map(letter_to_int) # New column mapping letter keys to numbers
data=data.drop(columns=['key','key_base']) #Dropping the helper columns

n_unknown = (data['tempo'] == '?').sum() # Determining '?'s in tempo
print(f"Unknown tempo entries: {n_unknown}")
data['tempo'] = data['tempo'].replace('?', np.nan).astype(float) # converting '?' to NaNs
median_tempo = data['tempo'].median()
print(f"Median tempo (computed on known values): {median_tempo:.2f}")
data['tempo'].fillna(median_tempo, inplace=True) # Imputing NaNs with median.

neg_durs = (data['duration_ms'] < 0).sum() # Finding invalid duration (negative).
print(f"Negative duration entries: {neg_durs}")
data.loc[data['duration_ms'] < 0, 'duration_ms'] = np.nan # converting invalid vals to NaNs.
median_dur = data['duration_ms'].median() 
print(f"Median duration (ms): {median_dur:.0f}")
data['duration_ms'].fillna(median_dur, inplace=True) # Imputing NaNs with median.


month_map = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
}
extract = data['obtained_date'].str.extract(
    r'(?P<day>\d{1,2})[^\dA-Za-z]*(?P<mon>[A-Za-z]{3,})'
)
data['obtained_day'] = pd.to_numeric(extract['day'], errors='coerce')
data['obtained_month_num'] = (
    extract['mon']
     .str[:3]                       
     .str.title()                  
     .map(month_map)                
)

print("Parsed days:", data['obtained_day'].notna().sum())
print("Parsed months:", data['obtained_month_num'].notna().sum())

# Impute any remaining NaNs with medians
data['obtained_day'].fillna(data['obtained_day'].median(), inplace=True)
data['obtained_month_num'].fillna(
    data['obtained_month_num'].median(), inplace=True
)
data.drop(columns=['obtained_date'], inplace=True) # Dropping original column.

print(data.head())

#%% Train/Test Split
test_df_list = []
train_df_list = []

for genre in data['music_genre'].unique():
    genre_subset = data[data['music_genre'] == genre]
    test_samples = genre_subset.sample(n=500, random_state=42) # Selecting 500 songs for test set.
    train_samples = genre_subset.drop(test_samples.index) # The other 4500 go to the train set.
    test_df_list.append(test_samples)
    train_df_list.append(train_samples)

train_df = pd.concat(train_df_list).sample(frac=1, random_state=42)
test_df = pd.concat(test_df_list).sample(frac=1, random_state=42)
print("Shape of Training Set: ",train_df.shape)
print("Shape of Test Set: ",test_df.shape)

X_train = train_df.drop(columns=['music_genre'])
X_train = X_train.drop(columns=['artist_name', 'track_name'], errors='ignore')

X_test = test_df.drop(columns=['music_genre'])
X_test = X_test.drop(columns=['artist_name', 'track_name'], errors='ignore')

y_train = train_df['music_genre']
y_test = test_df['music_genre']

le = LabelEncoder()
# Label encoding y
y_train_enc = le.fit_transform(y_train)   
y_test_enc  = le.transform(y_test)

# scaling input for PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#%% Dimensionality Reduction 

pca = PCA(whiten=True, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_pca[:,0], y=X_pca[:,1],
    hue=y_train, palette='tab10', alpha=0.6, legend='brief'
)
plt.title(f'PCA (PC1 {pca.explained_variance_ratio_[0]*100:.1f}%  PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()

explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_var), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.axhline(0.8, color='red', linestyle='--', label='80% Variance')
plt.legend()
plt.show()

pca_lower = PCA(n_components=9, whiten=True, random_state=42)
X_lower_pca = pca_lower.fit_transform(X_train_scaled)
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_lower_pca[:,0], y=X_lower_pca[:,1],
    hue=y_train, palette='tab10', alpha=0.6, legend='brief'
)
plt.title(f'PCA Lower (PC1 {pca_lower.explained_variance_ratio_[0]*100:.1f}%  PC2 {pca_lower.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()

#%% Determining Optimal Classification Model

X_train_pca = pca.transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

X_sub, _, y_sub, _ = train_test_split( # Taking a subset of training data to speed up search.
    X_train_pca,
    y_train_enc,
    train_size=5000,
    random_state=42,
    stratify=y_train_enc
)

# Testing 5 different models in a loop.
models = [
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("SVM (RBF)", SVC(kernel="rbf", probability=True, random_state=42)),
    ("SVM (Linear)", SVC(kernel="linear", probability=True, random_state=42)),
]

results_sub = {}
for name, clf in models:
    scores = cross_val_score(
        clf,
        X_sub,
        y_sub,
        cv=3,
        scoring="roc_auc_ovr",
        n_jobs=-1
    )
    results_sub[name] = (scores.mean(), scores.std())

for name, (m, s) in sorted(results_sub.items(), key=lambda x: -x[1][0]):
    print(f"{name:20s}  {m:.3f} ± {s:.3f}")
    
# Testing Perceptron
base = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
calib = CalibratedClassifierCV(base, method='sigmoid', cv=3)
scores = cross_val_score(
    calib,
    X_train_pca,
    y_train_enc,
    cv=5,
    scoring="roc_auc_ovr",
    n_jobs=-1
)
print(f"Perceptron CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

calib.fit(X_train_pca, y_train_enc)
y_proba = calib.predict_proba(X_test_pca)
perc_test_auc = roc_auc_score(y_test_enc, y_proba, multi_class='ovr', average='macro')
print(f"Perceptron Test AUC: {perc_test_auc:.3f}")

num_features = X_train_pca.shape[1]
num_classes  = len(le.classes_)

# Testing NN - basic version
model = Sequential([
    InputLayer(input_shape=(num_features,)),
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']     
)

model.fit(
    X_train_pca,
    y_train_enc,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

y_proba = model.predict(X_test_pca)   
nn_test_auc = roc_auc_score(
    y_test_enc,
    y_proba,
    multi_class='ovr',
    average='macro'
)
print(f"Basic NN AUC = {nn_test_auc:.3f}")

#%% Optimizing Parameters for NN
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split( # even smaller subset to speed up our search
    X_sub, y_sub, 
    test_size=0.2, 
    stratify=y_sub, 
    random_state=42
)

param_grid = { # param search grid
    "units1":    [32, 64, 128, 256],
    "dropout1":  [0.2, 0.3, 0.5],
    "learning_rate": [1e-2, 1e-3, 1e-4]
}

results = []

for units1, dropout1, lr in itertools.product(
        param_grid["units1"], 
        param_grid["dropout1"], 
        param_grid["learning_rate"]
    ):
    model = Sequential([
        InputLayer(input_shape=(X_sub.shape[1],)),
        Dense(units1, activation='relu'),
        Dropout(dropout1),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy'
    )
    
    model.fit(
        X_train_sub, y_train_sub,
        epochs=10,            
        batch_size=32,
        verbose=0
    )
    y_val_proba = model.predict(X_val_sub)
    
    auc = roc_auc_score(
        y_val_sub, 
        y_val_proba, 
        multi_class='ovr', 
        average='macro'
    )
    print(f"units={units1}, drop={dropout1}, lr={lr} → val AUC {auc:.4f}")
    results.append((auc, units1, dropout1, lr))

best_auc, best_units1, best_dropout1, best_lr = max(results, key=lambda x: x[0])
print("\nBest on subsample:",
      f"AUC={best_auc:.4f}, units={best_units1}, dropout={best_dropout1}, lr={best_lr}")

best_model = Sequential([
    InputLayer(input_shape=(X_sub.shape[1],)),
    Dense(best_units1, activation='relu'),
    Dropout(best_dropout1),
    Dense(num_classes, activation='softmax')
])
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
    loss='sparse_categorical_crossentropy'
)
best_model.fit(
    X_sub, y_sub,
    epochs=20,
    batch_size=32,
    verbose=1
)

y_test_proba = best_model.predict(X_test_pca)
test_auc = roc_auc_score(y_test_enc, y_test_proba, multi_class='ovr', average='macro')
print(f"Final test AUC (NN) = {test_auc:.4f}")

#%% Optimizing Parameters for SVC Lin
base_svc = SVC(kernel="linear", probability=True, random_state=42)

param_grid = {
    # only C matters for linear
    'C': np.logspace(-2, 2, 10)
}

search = GridSearchCV(
    estimator=base_svc,
    param_grid=param_grid,
    scoring="roc_auc_ovr",
    cv=5,
    n_jobs=-1,
    verbose=1
)

search.fit(X_sub, y_sub)
print("Best C:",      search.best_params_['C'])
print("Best CV AUC:", search.best_score_)

#%% Optimized Neural Network 
model = Sequential([
    InputLayer(input_shape=(num_features,)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy'
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history = model.fit(
    X_train_pca, y_train_enc,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

y_nn_proba = model.predict(X_test_pca)
final_nn_auc = roc_auc_score(y_test_enc, y_nn_proba, multi_class='ovr', average='macro')
print(f"Final Tuned NN AUC = {final_nn_auc:.4f}")

#%% Final AUC & ROC

y_test_bin = label_binarize(y_test_enc, classes=range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_nn_proba[:, i])
    roc_auc[i] = sk_auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = sk_auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(
        fpr[i], tpr[i],
        lw=1,
        label=f"Class {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot(
    fpr["macro"], tpr["macro"],
    color='black', lw=2, linestyle='--',
    label=f"Macro‑avg (AUC = {roc_auc['macro']:.2f})"
)

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


y_pred = np.argmax(y_nn_proba, axis=1)   
cm = confusion_matrix(y_test_enc, y_pred) #confusion matrix for all classes.

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

