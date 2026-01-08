"""
Script untuk training model Keras food waste prediction
Output: model terlatih (.keras) dan scaler (.pkl)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Keras imports
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def load_and_preprocess_data(csv_path):
    """Load CSV dan preprocess data untuk training"""
    
    print("[INFO] Loading dataset...")
    df = pd.read_csv('prototype/data/food_waste_dataset2.csv')
    print(f"[INFO] Loaded {len(df)} records")
    
    # Categorical columns to encode
    categorical_cols = ['product_category', 'event_type', 'store_location', 'storage_condition']
    
    # Create label encoders
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"[INFO] Encoded {col}: {len(le.classes_)} categories")
    
    # Feature columns
    feature_cols = [
        'product_category_encoded',
        'event_type_encoded', 
        'store_location_encoded',
        'storage_condition_encoded',
        'shelf_life_days',
        'days_before_expiry',
        'initial_stock',
        'expected_daily_sales',
        'price_per_unit',
        'supplier_reliability',
        'temperature_deviation',
        'historical_waste_rate',
        'promotion_active',
        'day_of_week',
        'month',
        'weather_score'
    ]
    
    X = df[feature_cols].values
    y = df['waste_percentage'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders, feature_cols


def build_model(input_dim):
    """Build Keras neural network model"""
    
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Layer 1
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 2
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Layer 4
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Layer 5
        Dense(16, activation='relu'),
        
        # Output layer (waste percentage 0-1)
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def train_model(X, y, epochs=250, batch_size=32):
    """Train the model with callbacks for better performance"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"\n[INFO] Data Split:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    # Build model
    model = build_model(X_train.shape[1])
    
    print("\n[INFO] Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'food_waste_model_best2.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\n[INFO] Starting training for {epochs} epochs...")
    print("=" * 60)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.6f}")
    
    # Predictions analysis
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # R-squared
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f"R-squared Score: {r2_score:.4f}")
    
    # Mean Absolute Percentage Error
    # mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    # print(f"MAPE: {mape:.2f}%")
    
    # Accuracy within tolerance
    # tolerance = 0.05 # 5% tolerance
    # accurate_preds = np.sum(np.abs(y_test - y_pred) <= tolerance)
    # accuracy = accurate_preds / len(y_test) * 100
    # print(f"Accuracy (within {tolerance*100}% tolerance): {accuracy:.2f}%")
    
    return model, history, (X_test, y_test)


def save_artifacts(model, scaler, label_encoders, feature_cols):
    """Save model and preprocessing artifacts"""
    
    # Save model
    model.save('food_waste_model2.keras')
    print("\n[SAVED] Model saved to: food_waste_model.keras")
    
    # Save scaler and encoders
    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols
    }
    
    with open('model_artifacts2.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("[SAVED] Artifacts saved to: model_artifacts.pkl")


if __name__ == "__main__":
    print("=" * 60)
    print("FOOD WASTE PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    csv_path = "protoype/data/food_waste_dataset2.csv"
    X, y, scaler, label_encoders, feature_cols = load_and_preprocess_data(csv_path)
    
    # Train model
    model, history, test_data = train_model(
        X, y, 
        epochs=250,  # 250 epochs untuk training yang lebih baik
        batch_size=32
    )
    
    # Save all artifacts
    save_artifacts(model, scaler, label_encoders, feature_cols)
    
    # Training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nFiles generated:")
    print("  1. food_waste_model.keras - Trained Keras model")
    print("  2. food_waste_model_best.keras - Best model checkpoint")
    print("  3. model_artifacts.pkl - Scaler and label encoders")
