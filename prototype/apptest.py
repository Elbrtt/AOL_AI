"""
Flask MVP - Smart Food Waste Prediction System
Website untuk memprediksi food waste dan memberikan rekomendasi mitigasi
Bekerja dengan dataset: enhanced_smart_waste_dataset_v1.csv
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from tensorflow import keras
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = 'smart_waste_prediction_key'

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class SmartWasteAnalyzer:
    """Service untuk analisis prediksi food waste dan spoilage"""
    
    def __init__(self, model_path='prototype/proto_model/waste_prediction_model.keras', artifacts_path='prototype/proto_model/model_artifacts.pkl'):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_cols = None
        self.is_loaded = False
        
        try:
            self.model = keras.models.load_model(model_path)
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            self.scaler = artifacts['scaler']
            self.label_encoders = artifacts['label_encoders']
            self.feature_cols = artifacts['feature_cols']
            self.is_loaded = True
            print("[INFO] Model berhasil dimuat")
        except Exception as e:
            print(f"[WARNING] Model belum tersedia: {e}")
            print("[INFO] Jalankan prototype_training.py terlebih dahulu")
    
    def predict_spoilage_chance(self, row):
        """Prediksi spoilage chance menggunakan model neural network"""
        if not self.is_loaded:
            print("[ERROR] Model not loaded!")
            return None
        
        try:
            features_for_prediction = {}
            
            # Encode Category
            if 'category' in self.label_encoders and 'Category' in row:
                category_val = row['Category']
                if category_val in self.label_encoders['category'].classes_:
                    features_for_prediction['category_encoded'] = self.label_encoders['category'].transform([category_val])[0]
                else:
                    features_for_prediction['category_encoded'] = 0
            else:
                features_for_prediction['category_encoded'] = 0
            
            # Encode StoreLocation
            if 'store_location' in self.label_encoders and 'StoreLocation' in row:
                location_val = row['StoreLocation']
                if location_val in self.label_encoders['store_location'].classes_:
                    features_for_prediction['store_location_encoded'] = self.label_encoders['store_location'].transform([location_val])[0]
                else:
                    features_for_prediction['store_location_encoded'] = 0
            else:
                features_for_prediction['store_location_encoded'] = 0
            
            # Encode Brand
            if 'brand' in self.label_encoders and 'Brand' in row:
                brand_val = row['Brand']
                if brand_val in self.label_encoders['brand'].classes_:
                    features_for_prediction['brand_encoded'] = self.label_encoders['brand'].transform([brand_val])[0]
                else:
                    features_for_prediction['brand_encoded'] = 0
            else:
                features_for_prediction['brand_encoded'] = 0
            
            # Encode TemperatureSensitive
            if 'temperaturesensitive' in self.label_encoders and 'TemperatureSensitive' in row:
                temp_val = bool(row['TemperatureSensitive'])
                if temp_val in self.label_encoders['temperaturesensitive'].classes_:
                    features_for_prediction['temp_encoded'] = self.label_encoders['temperaturesensitive'].transform([temp_val])[0]
                else:
                    features_for_prediction['temp_encoded'] = 0
            else:
                features_for_prediction['temp_encoded'] = 0

            # Add numerical features
            features_for_prediction['StockQty'] = float(row.get('StockQty', 0))
            features_for_prediction['DaysUntilExpiry'] = float(row.get('DaysUntilExpiry', 0))
            features_for_prediction['DailySaleAvg'] = float(row.get('DailySaleAvg', 0))
            features_for_prediction['DistanceToNearestStore'] = float(row.get('DistanceToNearestStore', 0))
            features_for_prediction['AvgDailySaleInNearbyStores'] = float(row.get('AvgDailySaleInNearbyStores', 0))
            
            # Create DataFrame with encoded values in correct order
            features_df = pd.DataFrame([features_for_prediction])
            
            # Select features in the order model expects
            X = features_df[self.feature_cols].fillna(0).values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled, verbose=0)
            spoilage_chance = float(prediction[0][0])
            
            # Clamp antara 0 dan 1
            spoilage_chance = max(0, min(1, spoilage_chance))
            
            return spoilage_chance
        except Exception as e:
            print(f"[ERROR] Error predicting spoilage: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_spoilage_risk_level(self, spoilage_chance, days_until_expiry, is_spoiled):
        """Kategorisasi level risiko spoilage"""
        if is_spoiled:
            return {'level': 'KRITIS', 'color': 'critical', 'icon': '✗', 'desc': 'Produk Sudah Rusak'}
        elif spoilage_chance >= 0.8:
            return {'level': 'KRITIS', 'color': 'danger', 'icon': '⚠', 'desc': 'Risiko Spoilage Sangat Tinggi'}
        elif spoilage_chance >= 0.6:
            return {'level': 'TINGGI', 'color': 'danger', 'icon': '⚠', 'desc': 'Risiko Spoilage Tinggi'}
        elif spoilage_chance >= 0.3:
            return {'level': 'SEDANG', 'color': 'warning', 'icon': '⚠', 'desc': 'Risiko Spoilage Sedang'}
        else:
            return {'level': 'RENDAH', 'color': 'success', 'icon': '✓', 'desc': 'Risiko Spoilage Rendah'}
    
    def calculate_waste_potential(self, stock_qty, spoilage_chance, days_until_expiry, daily_sale_avg):
        """Hitung potensi waste (kg atau unit)"""
        if days_until_expiry <= 0:
            return stock_qty  # Sudah expired, semua potential waste
        
        # Hitung berapa unit yang mungkin terjual sebelum expiry
        potential_sales = daily_sale_avg * days_until_expiry
        unsold = max(0, stock_qty - potential_sales)
        
        # Kalikan dengan spoilage chance
        potential_waste = unsold * spoilage_chance
        return round(potential_waste, 2)
    
    def generate_waste_recommendations(self, row, predicted_spoilage_chance):
        """Generate rekomendasi berdasarkan predicted spoilage chance dari model"""
        recommendations = []
        
        days_until_expiry = float(row.get('DaysUntilExpiry', 0))
        is_spoiled = row.get('IsSpoiled', False)
        stock_qty = int(row.get('StockQty', 0))
        
        # Rekomendasi berdasarkan predicted spoilage chance
        if predicted_spoilage_chance >= 0.8:
            recommendations.append({
                'prioritas': 'KRITIS',
                'kategori': 'Donasi/Diskon Urgent',
                'pesan': f"Prediksi spoilage sangat tinggi ({predicted_spoilage_chance*100:.1f}%). Produk harus segera ditindaklanjuti",
                'aksi': 'Segera donasikan atau berikan diskon besar-besaran untuk move inventory'
            })
        elif predicted_spoilage_chance >= 0.6:
            recommendations.append({
                'prioritas': 'TINGGI',
                'kategori': 'Flash Sale / Diskon',
                'pesan': f"Prediksi spoilage tinggi ({predicted_spoilage_chance*100:.1f}%). Stok: {stock_qty} unit",
                'aksi': 'Tawarkan diskon 20-40% dan promosikan untuk mempercepat penjualan'
            })
        elif predicted_spoilage_chance >= 0.3:
            recommendations.append({
                'prioritas': 'SEDANG',
                'kategori': 'Monitor & Promosi',
                'pesan': f"Spoilage sedang ({predicted_spoilage_chance*100:.1f}%). Monitor kondisi produk",
                'aksi': 'Pantau kondisi, cek regular stock, dan pertimbangkan promosi ringan'
            })
        else:
            recommendations.append({
                'prioritas': 'RENDAH',
                'kategori': 'Maintain Stock',
                'pesan': f"Spoilage rendah ({predicted_spoilage_chance*100:.1f}%). Produk stabil",
                'aksi': 'Maintain stock levels dan monitor sales trend secara normal'
            })
        
        # Rekomendasi berdasarkan hari sampai expiry
        if days_until_expiry <= 0:
            recommendations.append({
                'prioritas': 'KRITIS',
                'kategori': 'Produk Expired',
                'pesan': 'Produk sudah expired! Harus segera dimusnahkan.',
                'aksi': 'Dokumentasikan dan lakukan disposal sesuai prosedur'
            })
        elif days_until_expiry < 3:
            recommendations.append({
                'prioritas': 'KRITIS',
                'kategori': 'Expiry Sangat Dekat',
                'pesan': f'Tinggal {days_until_expiry:.1f} hari sampai expiry. Situasi darurat!',
                'aksi': 'Pindahkan ke rak paling depan, berikan special offer'
            })
        elif days_until_expiry < 7:
            recommendations.append({
                'prioritas': 'TINGGI',
                'kategori': 'Expiry Dekat',
                'pesan': f'Tinggal {days_until_expiry:.1f} hari sampai expiry',
                'aksi': 'Promosi khusus, bundling, atau pertimbangkan donasi'
            })
        
        return recommendations
    
    def analyze_single(self, row):
        """Analisis satu produk dari dataset"""
        try:
            item_name = row.get('ItemName', 'Unknown')
            brand = row.get('Brand', 'N/A')
            category = row.get('Category', 'Unknown')
            stock_qty = int(row.get('StockQty', 0))
            days_until_expiry = float(row.get('DaysUntilExpiry', 0))
            daily_sale_avg = float(row.get('DailySaleAvg', 0))
            is_spoiled = row.get('IsSpoiled', False)
            on_promotion = row.get('OnPromotion', 'No')
            temperature_sensitive = row.get('TemperatureSensitive', False)
            store_location = row.get('StoreLocation', 'Unknown')
            store_id = row.get('StoreID', 'Unknown')
            
            predicted_spoilage_chance = self.predict_spoilage_chance(row)
            if predicted_spoilage_chance is None:
                return {'error': 'Model prediction failed'}
            
            # Hitung metrics
            potential_waste = self.calculate_waste_potential(stock_qty, predicted_spoilage_chance, days_until_expiry, daily_sale_avg)
            days_to_expiry = max(0, days_until_expiry)
            
            # Hitung berapa unit yang akan terjual sebelum expiry
            units_will_sell = min(stock_qty, int(daily_sale_avg * days_to_expiry))
            units_waste = stock_qty - units_will_sell
            waste_percentage = (units_waste / stock_qty * 100) if stock_qty > 0 else 0
            
            # Risk assessment
            risk = self.get_spoilage_risk_level(predicted_spoilage_chance, days_until_expiry, is_spoiled)
            
            return {
                'item_name': item_name,
                'brand': brand,
                'category': category,
                'store_location': store_location,
                'store_id': store_id,
                'stock_qty': stock_qty,
                'spoilage_chance': round(predicted_spoilage_chance, 4),
                'spoilage_chance_pct': round(predicted_spoilage_chance * 100, 1),
                'days_until_expiry': round(days_until_expiry, 1),
                'daily_sale_avg': round(daily_sale_avg, 2),
                'is_spoiled': is_spoiled,
                'on_promotion': on_promotion,
                'temperature_sensitive': temperature_sensitive,
                'potential_waste': potential_waste,
                'units_will_sell': units_will_sell,
                'units_waste': units_waste,
                'waste_percentage': round(waste_percentage, 1),
                'risk_level': risk,
                'recommendations': self.generate_waste_recommendations(row, predicted_spoilage_chance)
            }
        except Exception as e:
            return {'error': f'Error analyzing product: {str(e)}'}
    
    def analyze_from_csv(self, csv_path):
        """Analisis batch dari file CSV"""
        try:
            df = pd.read_csv(csv_path)
            
            # Convert TemperatureSensitive to boolean
            if 'TemperatureSensitive' in df.columns:
                df['TemperatureSensitive'] = df['TemperatureSensitive'].astype(bool)
            
            results = []
            
            required_cols = ['ItemName', 'Category', 'Brand', 'StoreLocation', 'StockQty', 'DaysUntilExpiry', 'DailySaleAvg', 'DistanceToNearestStore', 'AvgDailySaleInNearbyStores','TemperatureSensitive']
            
            # Check which required columns exist
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                return {'error': f'Kolom tidak ditemukan: {", ".join(missing)}. CSV harus memiliki kolom: {", ".join(required_cols)}'}
            
            # Analyze each product
            total_potential_waste = 0
            risk_counts = {'RENDAH': 0, 'SEDANG': 0, 'TINGGI': 0, 'KRITIS': 0}
            spoiled_items = 0
            
            for idx, row in df.iterrows():
                result = self.analyze_single(row)
                if result and 'error' not in result:
                    result['nomor'] = idx + 1
                    results.append(result)
                    total_potential_waste += result['potential_waste']
                    risk_counts[result['risk_level']['level']] += 1
                    if result['is_spoiled']:
                        spoiled_items += 1
            
            # Summary statistics
            avg_spoilage_chance = np.mean([r['spoilage_chance'] for r in results]) if results else 0
            avg_waste_pct = np.mean([r['waste_percentage'] for r in results]) if results else 0
            
            return {
                'success': True,
                'total_products': len(results),
                'hasil_analisis': results,
                'ringkasan': {
                    'total_potential_waste_units': round(total_potential_waste, 2),
                    'avg_spoilage_chance': round(avg_spoilage_chance, 4),
                    'avg_waste_percentage': round(avg_waste_pct, 1),
                    'already_spoiled_items': spoiled_items,
                    'distribusi_risiko': risk_counts
                }
            }
        except Exception as e:
            return {'error': f'Gagal memproses CSV: {str(e)}'}


# Initialize analyzer
waste_analyzer = SmartWasteAnalyzer()


@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index2.html', model_loaded=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle upload CSV dan analisis"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File harus berformat CSV'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Analisis
    results = waste_analyzer.analyze_from_csv(filepath)
    
    # Hapus file setelah diproses
    os.remove(filepath)
    
    if 'error' in results:
        return jsonify(results), 400
    
    return jsonify(results)


@app.route('/template')
def template_info():
    """Provide template info"""
    template_info = {
        'kolom_wajib': [
            {'nama': 'ItemName', 'tipe': 'text', 'contoh': 'Bingo Mad Angles', 'keterangan': 'Nama produk'},
            {'nama': 'Category', 'tipe': 'text', 'contoh': 'Snacks', 'keterangan': 'Kategori produk'},
            {'nama': 'StockQty', 'tipe': 'number', 'contoh': '92', 'keterangan': 'Jumlah stok saat ini'},
            {'nama': 'DaysUntilExpiry', 'tipe': 'decimal', 'contoh': '0.5', 'keterangan': 'Hari sampai kadaluarsa'},
            {'nama': 'DailySaleAvg', 'tipe': 'decimal', 'contoh': '12.1', 'keterangan': 'Rata-rata penjualan harian'},
            {'nama': 'Brand', 'tipe': 'text', 'contoh': 'ITC', 'keterangan': 'Brand produk'},
            {'nama': 'StoreLocation', 'tipe': 'text', 'contoh': 'Chennai', 'keterangan': 'Lokasi toko'},
            {'nama': 'DistanceToNearestStore', 'tipe': 'decimal', 'contoh': '5.2', 'keterangan': 'Jarak ke toko terdekat'},
            {'nama': 'AvgDailySaleInNearbyStores', 'tipe': 'decimal', 'contoh': '10.3', 'keterangan': 'Rata-rata penjualan harian di toko terdekat'},
            {'nama': 'TemperatureSensitive', 'tipe': 'boolean', 'contoh': 'False', 'keterangan': 'Sensitif terhadap suhu'}
        ],
        'kolom_opsional': [
            {'nama': 'StoreID', 'tipe': 'text', 'contoh': 'Chennai_0', 'keterangan': 'ID toko unik'},
            {'nama': 'IsSpoiled', 'tipe': 'boolean', 'contoh': 'False', 'keterangan': 'Apakah produk sudah rusak'},
            {'nama': 'OnPromotion', 'tipe': 'text', 'contoh': 'No', 'keterangan': 'Apakah sedang promo'}
        ]
    }
    return jsonify(template_info)


if __name__ == '__main__':
    print("=" * 70)
    print("SMART FOOD WASTE PREDICTION SYSTEM")
    print("=" * 70)
    print("Sistem prediksi food waste dan spoilage dengan rekomendasi mitigasi")
    print("Dataset: enhanced_smart_waste_dataset_v1.csv")
    print("Buka browser: http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)