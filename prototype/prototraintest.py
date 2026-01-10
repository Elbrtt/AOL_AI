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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_model_artifacts(csv_path='prototype/data/enhanced_smart_waste_dataset_v1.csv'):  
    print("[INFO] Loading smart waste dataset...")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[SUCCESS] Dataset loaded: {csv_path}")
    else:
        print(f"[ERROR] Dataset file tidak ditemukan: {csv_path}")
        print("[INFO] Pastikan enhanced_smart_waste_dataset_v1.csv ada di direktori root")
        return
    
    print(f"[INFO] Loaded {len(df)} records")
    
    # Display basic dataset info
    print(f"[INFO] Dataset columns: {df.columns.tolist()}")
    print(f"[INFO] Dataset shape: {df.shape}")
    
    # Convert TemperatureSensitive to boolean
    if 'TemperatureSensitive' in df.columns:
        df['TemperatureSensitive'] = df['TemperatureSensitive'].astype(bool)
        print(f"[INFO] Converted TemperatureSensitive to boolean")
    
    label_encoders = {}
    
    # Encode Category column
    if 'Category' in df.columns:
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['Category'])
        label_encoders['category'] = le
        print(f"[INFO] Encoded category: {len(le.classes_)} categories")
        # Display sample mapping
        if len(le.classes_) <= 10:
            print(f"[DEBUG] Category mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            print(f"[DEBUG] Category mapping sample (first 10): {dict(zip(le.classes_[:10], le.transform(le.classes_[:10])))}")
    
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

    # Encode Temp column
    if 'TemperatureSensitive' in df.columns:
        le = LabelEncoder()
        df['temp_encoded'] = le.fit_transform(df['TemperatureSensitive'])
        label_encoders['temperaturesensitive'] = le
        print(f"[INFO] Encoded temperaturesensitive: {len(le.classes_)} categories")
    
    feature_cols = [
        'category_encoded',
        'store_location_encoded',
        'brand_encoded',
        'StockQty',
        'DaysUntilExpiry',
        'DailySaleAvg',
        'DistanceToNearestStore',
        'AvgDailySaleInNearbyStores',
        'temp_encoded'
    ]
    
    # Check available columns and filter feature_cols
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[WARNING] Kolom tidak tersedia: {missing_cols}, akan dilewatkan")
        feature_cols = available_cols
    
    print(f"[INFO] Features to use: {feature_cols}")
    
    try:
        # Prepare features and target
        X = df[feature_cols].fillna(0).values
        
        # Target: SpoilageChance (0-1 scale) - what the model will predict
        if 'SpoilageChance' not in df.columns:
            print("[ERROR] Kolom target 'SpoilageChance' tidak ditemukan dalam dataset")
            print(f"[INFO] Kolom yang tersedia: {df.columns.tolist()}")
            return
            
        y = df['SpoilageChance'].fillna(0).values
        
        print(f"[INFO] Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[INFO] Target: SpoilageChance (predicting from {X.shape[1]} features)")
        print(f"[INFO] Target range: {y.min():.4f} - {y.max():.4f}")
        print(f"[INFO] Target mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        # Check for target distribution
        print(f"[INFO] Target distribution:")
        print(f"  - Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")
        print(f"  - Samples with SpoilageChance > 0.5: {np.sum(y > 0.5)}")
        print(f"  - Samples with SpoilageChance > 0.7: {np.sum(y > 0.7)}")
        print(f"  - Samples with SpoilageChance > 0.9: {np.sum(y > 0.9)}")
        
    except Exception as e:
        print(f"[ERROR] Error preparing features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Scale features menggunakan semua data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build and train model
    print("\n" + "=" * 50)
    print("[INFO] Building neural network model...")
    print("=" * 50)
    
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
        metrics=['mae', 'mse']
    )
    
    # Display model summary
    print("\n[INFO] Model Architecture:")
    print("=" * 50)
    model.summary()
    print("=" * 50)
    
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=25,
            restore_best_weights=True,
            min_delta=0.0001,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n[INFO] Training model dengan semua data waste dataset...")
    print(f"[INFO] Menggunakan {X.shape[0]} sampel untuk training")
    
    history = model.fit(
        X_scaled, 
        y, 
        epochs=300, 
        batch_size=16, 
        verbose=1, 
        validation_split=0.1,
        callbacks=callbacks
    )
    
    train_loss, train_mae, train_mse = model.evaluate(X_scaled, y, verbose=0)
    
    # Predictions analysis pada semua data
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    
    # Calculate R-squared pada training data
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
 
    # Calculate MAPE (avoid division by zero)
    mask = y != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
    else:
        mape = None
    
    # Additional metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate correlation
    correlation = np.corrcoef(y, y_pred)[0, 1]
    # print(f"  Correlation (Actual vs Predicted): {correlation:.4f}")
    
    # Save artifacts
    os.makedirs('prototype/proto_model', exist_ok=True)
    
    # Save model
    model.save('prototype/proto_model/waste_prediction_model.keras')
    print(f"\n[SAVED] Model saved to: prototype/proto_model/waste_prediction_model.keras")
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'feature_names': feature_cols,
        'model_metrics': {
            'training_mse': float(train_mse),
            'training_mae': float(train_mae),
            'training_rmse': float(np.sqrt(train_mse)),
            'r2_score': float(r2_score),
            'mape': float(mape) if mape is not None else None,
            'correlation': float(correlation),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1])
        },
        'data_summary': {
            'target_min': float(y.min()),
            'target_max': float(y.max()),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'n_categories': len(label_encoders.get('category', LabelEncoder()).classes_) if 'category' in label_encoders else 0,
            'n_locations': len(label_encoders.get('store_location', LabelEncoder()).classes_) if 'store_location' in label_encoders else 0,
            'n_brands': len(label_encoders.get('brand', LabelEncoder()).classes_) if 'brand' in label_encoders else 0
        }
    }
    
    with open('prototype/proto_model/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("[SAVED] Artifacts saved to: prototype/proto_model/model_artifacts.pkl")
    
    # Save training history for analysis
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('prototype/proto_model/training_history.csv', index=False)
    print("[SAVED] Training history saved to: prototype/proto_model/training_history.csv")
    
    # Save predictions vs actual untuk semua data
    all_predictions = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'difference': y - y_pred,
        'abs_error': np.abs(y - y_pred),
        'prediction_accuracy': 1 - np.abs(y - y_pred)  # 1 - absolute error
    })
    
    all_predictions.to_csv('prototype/proto_model/all_predictions.csv', index=False)
    print("[SAVED] All predictions saved to: prototype/proto_model/all_predictions.csv")
    
    # Summary statistics
    error_stats = {
        'mean_absolute_error': float(np.mean(np.abs(y - y_pred))),
        'max_absolute_error': float(np.max(np.abs(y - y_pred))),
        'min_absolute_error': float(np.min(np.abs(y - y_pred))),
        'std_absolute_error': float(np.std(np.abs(y - y_pred))),
        'mean_prediction_accuracy': float(np.mean(1 - np.abs(y - y_pred))) * 100
    }
    
    error_stats_df = pd.DataFrame([error_stats])
    error_stats_df.to_csv('prototype/proto_model/error_statistics.csv', index=False)
    print("[SAVED] Error statistics saved to: prototype/proto_model/error_statistics.csv")
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model trained on ALL {len(df)} waste data samples")
    print(f"Input Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"Target Variable: SpoilageChance (0-1 probability)")
    print(f"\nModel Performance (on all training data):")
    print(f"  - R² Score: {r2_score:.4f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - Correlation: {correlation:.4f}")
    if mape is not None:
        print(f"  - MAPE: {mape:.2f}%")
    print(f"  - Mean Prediction Accuracy: {error_stats['mean_prediction_accuracy']:.2f}%")
    print("=" * 70)
    
    # Display sample predictions
    print("\n[INFO] Sample predictions (first 10 samples):")
    sample_indices = np.random.choice(len(y), min(10, len(y)), replace=False)
    for i, idx in enumerate(sample_indices):
        print(f"  Sample {i+1}: Actual={y[idx]:.4f}, Predicted={y_pred[idx]:.4f}, "
              f"Diff={abs(y[idx]-y_pred[idx]):.4f}, "
              f"Acc={(1-abs(y[idx]-y_pred[idx]))*100:.1f}%")
    
    # Display worst predictions
    print("\n[INFO] Worst predictions (highest error, first 5 samples):")
    errors = np.abs(y - y_pred)
    worst_indices = np.argsort(errors)[-5:][::-1]
    for i, idx in enumerate(worst_indices):
        print(f"  Worst {i+1}: Actual={y[idx]:.4f}, Predicted={y_pred[idx]:.4f}, "
              f"Error={errors[idx]:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("SMART WASTE PREDICTION MODEL TRAINING")
    print("TRAINING MODE: ALL DATA (No Test Split)")
    print("=" * 70)
    
    create_model_artifacts()