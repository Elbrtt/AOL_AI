"""
Script untuk training waste prediction model dari smart waste dataset
Menggunakan data actual dari enhanced_smart_waste_dataset_v1.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def create_model_artifacts(csv_path='prototype/data/enhanced_smart_waste_dataset_v1.csv'):
    """Create model artifacts from actual waste dataset"""
    
    print("[INFO] Loading smart waste dataset...")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[SUCCESS] Dataset loaded: {csv_path}")
    else:
        print(f"[ERROR] Dataset file tidak ditemukan: {csv_path}")
        print("[INFO] Pastikan enhanced_smart_waste_dataset_v1.csv ada di direktori root")
        return
    
    print(f"[INFO] Loaded {len(df)} records")
    
    label_encoders = {}
    
    # Encode Category column
    if 'Category' in df.columns:
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['Category'])
        label_encoders['category'] = le
        print(f"[INFO] Encoded category: {len(le.classes_)} categories")
    
    
    # Encode StoreLocation column
    if 'StoreLocation' in df.columns:
        le = LabelEncoder()
        df['store_location_encoded'] = le.fit_transform(df['StoreLocation'])
        label_encoders['store_location'] = le
        print(f"[INFO] Encoded store_location: {len(le.classes_)} categories")
    
    # Encode Brand column
    if 'Brand' in df.columns:
        le = LabelEncoder()
        df['brand_encoded'] = le.fit_transform(df['Brand'])
        label_encoders['brand'] = le
        print(f"[INFO] Encoded brand: {len(le.classes_)} categories")
    
    feature_cols = [
        'category_encoded',
        'store_location_encoded',
        'brand_encoded',
        'StockQty',
        'DaysUntilExpiry',
        'DailySaleAvg',
        'DistanceToNearestStore',
        'AvgDailySaleInNearbyStores'
    ]
    
    # Check available columns and filter feature_cols
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[WARNING] Kolom tidak tersedia: {missing_cols}, akan dilewatkan")
        feature_cols = available_cols
    
    try:
        X = df[feature_cols].fillna(0).values
        # Target: SpoilageChance (0-1 scale) - what the model will predict
        y = df['SpoilageChance'].fillna(0).values
        print(f"[INFO] Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[INFO] Target: SpoilageChance (predicting from {X.shape[1]} features)")
        print(f"[INFO] Target range: {y.min():.4f} - {y.max():.4f}")
    except Exception as e:
        print(f"[ERROR] Error preparing features: {e}")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build and train model
    print("\n[INFO] Building neural network model...")
    model = Sequential([
        Input(shape=(len(feature_cols),)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Output: spoilage probability (0-1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("[INFO] Training model with waste dataset...")
    model.fit(X_scaled, y, epochs=100, batch_size=16, verbose=1, validation_split=0.2)
    
    # Save artifacts
    os.makedirs('prototype/proto_model', exist_ok=True)
    
    model.save('prototype/proto_model/waste_prediction_model.keras')
    print("[SAVED] Model saved to: prototype/proto_model/waste_prediction_model.keras")
    
    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }
    
    with open('prototype/proto_model/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("[SAVED] Artifacts saved to: prototype/proto_model/model_artifacts.pkl")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model trained on {len(df)} actual waste data samples")
    print(f"Input Features: {', '.join(feature_cols)}")
    print(f"Target Variable: SpoilageChance (0-1 probability)")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("SMART WASTE PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    create_model_artifacts()
