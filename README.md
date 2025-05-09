# Spotify Classification Project

This project tackles multi-class classification of song genres using a real-world dataset from Spotify. The goal was to preprocess the data, reduce dimensionality, train several classification models, and maximize AUC performance on unseen data.

### Data Preprocessing
- Initial Dataset: 50,005 songs × 18 features
- Null Values: 90 total (5 rows dropped → 50,000 remaining)

Fixes:
Converted invalid “tempo” values to NaN → imputed with median: 119.88
Replaced negative durations → imputed with median: 227,360 ms

Encoding:
mode: binary (Minor → 1, Major → 0)
key: extracted is_sharp and mapped note (A-G) to 0–6
obtained_date: parsed to numeric month/day, imputed where needed

Dropped: artist_name, track_name
Target: Encoded music_genre to integer labels
Scaling: Used StandardScaler on all features

### Dimensionality Reduction (PCA)
Whitening applied for unit variance + decorrelation
Explained Variance:
PC1: 23.7%
PC2: 8.6%

Top 9 PCs explain >80% variance
All components retained for classification
Genre clusters like Classical and Rap visibly separated in 2D PCA

![image](https://github.com/user-attachments/assets/81ea4595-3ee6-4bbb-8962-74afdd8f76aa)
![image](https://github.com/user-attachments/assets/bef063a5-1bf1-4718-a953-26ac3cce83cb)

### Model Selection & Tuning
Initial Comparison (AUC on Subset):

SVM (Linear)	0.908 ± 0.002
SVM (RBF)	0.907 ± 0.001
Random Forest	0.885 ± 0.002
AdaBoost	0.747 ± 0.013
Decision Tree	0.640 ± 0.007
Perceptron	0.839 ± 0.011
Neural Net (basic)	0.902

**Best Tuned Models:**

LinearSVC (best C=35.9) → AUC: 0.9082
Neural Network (256–128 layers, 0.3 dropout) → AUC: 0.9122

### Final Deep Neural Network

*Architecture*

Input → Dense(256, ReLU) → BN → Dropout(0.3)  
      → Dense(128, ReLU) → BN → Dropout(0.3)  
      → Output (Softmax)
Optimizer: Adam (lr=1e-3)

Loss: Sparse Categorical Crossentropy

Callbacks: EarlyStopping, ReduceLROnPlateau

Trained: 30 epochs, batch size 64, 10% validation

*Final Performance* Test Set AUC (macro-average): 0.9339
![image](https://github.com/user-attachments/assets/8bc2df49-268a-47bf-89d2-8eba773e11e8)



