from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from tensorflow import keras

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'food_waste_prediction_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class FoodWastePredictionService:
    def __init__(self, model_path='prototype/proto_model/food_waste_model.keras', artifacts_path='prototype/proto_model/model_artifacts.pkl'):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_cols = None
        self.is_loaded = False
        
        if os.path.exists(model_path) and os.path.exists(artifacts_path):
            try:
                self.model = keras.models.load_model(model_path)
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                self.scaler = artifacts['scaler']
                self.label_encoders = artifacts['label_encoders']
                self.feature_cols = artifacts['feature_cols']
                self.is_loaded = True
            except Exception as e:
                print(f"[ERROR] Gagal memuat model: {e}")

    def get_risk_level_indo(self, waste_pct):
        # Normalisasi: pastikan input dalam desimal (0.0 - 1.0)
        if waste_pct < 0.10: return {'level': 'RENDAH', 'color': 'success', 'icon': '✓'}
        elif waste_pct < 0.20: return {'level': 'SEDANG', 'color': 'warning', 'icon': '⚠'}
        elif waste_pct < 0.35: return {'level': 'TINGGI', 'color': 'danger', 'icon': '⚠'}
        return {'level': 'KRITIS', 'color': 'danger', 'icon': '✗'}

    def calculate_optimal_stock(self, input_data, waste_pct):
        # LOGIKA DIPERBAIKI: Menghitung kebutuhan stok berdasarkan sisa hari kadaluarsa
        expected_demand = input_data['expected_daily_sales'] * input_data['days_before_expiry']
        
        event_safety_factors = {
            'Normal': 1.1, 'Mudik_Lebaran': 0.5, 'Pasca_Mudik': 1.2, 
            'Ramadan': 1.2, 'Natal_Tahun_Baru': 1.2, 'Long_Weekend': 0.8,
            'Hari_Raya_Idul_Adha': 0.7, 'Back_to_School': 1.1, 'Promo_Besar': 1.4
        }
        
        factor = event_safety_factors.get(input_data['event_type'], 1.0)
        # Stok optimal adalah demand dikali faktor event, dikurangi potensi waste
        optimal = int(expected_demand * factor * (1 - waste_pct))
        
        # Selisih untuk rekomendasi: Positif berarti kelebihan stok (Overstock)
        selisih = input_data['initial_stock'] - optimal
        # Potensi hemat: Jika overstock, nilai uang yang terselamatkan dari potensi waste
        potensi_hemat = max(0, selisih * input_data['price_per_unit'])
        
        return {
            'stok_disarankan': max(0, optimal),
            'stok_saat_ini': input_data['initial_stock'],
            'selisih': abs(selisih),
            'aksi': 'Kurangi Pesanan' if selisih > 0 else 'Tambah/Pertahankan',
            'potensi_penghematan': potensi_hemat
        }

    def process_features(self, df):
        """Helper untuk transformasi data secara batch (lebih cepat daripada loop)"""
        processed_df = df.copy()
        
        # Proteksi: Jika kategori baru tidak ada di encoder, gunakan kategori pertama (default)
        for col, le in self.label_encoders.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                )
        
        # Pastikan urutan kolom sesuai saat training
        return self.scaler.transform(processed_df[self.feature_cols])

    def predict_batch(self, df):
        if not self.is_loaded:
            return None
        
        try:
            # 1. Preprocessing Batch
            X_scaled = self.process_features(df)
            
            # 2. Prediksi Batch (Jauh lebih cepat daripada loop satu-satu)
            preds = self.model.predict(X_scaled, verbose=0).flatten()
            preds = np.clip(preds, 0, 1) # Pastikan antara 0-100%
            
            results = []
            for i, (idx, row) in enumerate(df.iterrows()):
                waste_pct = float(preds[i])
                res = {
                    'nomor': i + 1,
                    'produk': row['product_category'],
                    'prediksi_waste_persen': round(waste_pct * 100, 2),
                    'prediksi_kerugian': int(row['initial_stock'] * waste_pct * row['price_per_unit']),
                    'level_risiko': self.get_risk_level_indo(waste_pct),
                    'rekomendasi_stok': self.calculate_optimal_stock(row, waste_pct)
                }
                results.append(res)
            
            return results
        except Exception as e:
            print(f"Error prediction: {e}")
            return None

# Routes
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan'}), 400
    
    file = request.files['file']
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Format harus CSV'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        # Validasi kolom
        required = ['product_category', 'initial_stock', 'price_per_unit', 'expected_daily_sales', 'days_before_expiry']
        if not all(col in df.columns for col in required):
            return jsonify({'error': 'Kolom CSV tidak lengkap'}), 400

        results = prediction_service.predict_batch(df)
        
        if results is None:
            return jsonify({'error': 'Gagal memproses prediksi'}), 500

        # Ringkasan Statistik
        summary = {
            'total_kerugian': sum(r['prediksi_kerugian'] for r in results),
            'total_potensi_hemat': sum(r['rekomendasi_stok']['potensi_penghematan'] for r in results),
            'rata_waste': round(sum(r['prediksi_waste_persen'] for r in results) / len(results), 2)
        }

        return jsonify({'success': True, 'hasil': results, 'ringkasan': summary})
    
    finally:
        if os.path.exists(filepath): os.remove(filepath)

# ... (Route lainnya tetap sama)