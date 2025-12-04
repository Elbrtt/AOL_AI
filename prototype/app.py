"""
Flask MVP - Food Waste Prediction System
Website untuk memprediksi food waste dan memberikan rekomendasi stok
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from tensorflow import keras

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = 'food_waste_prediction_secret_key'

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class FoodWastePredictionService:
    """Service untuk prediksi food waste dengan output Bahasa Indonesia"""
    
    def __init__(self, model_path='prototype/proto_model/food_waste_model.keras', artifacts_path='prototype/proto_model/model_artifacts.pkl'):
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
            print("[INFO] Jalankan train_model.py terlebih dahulu")
    
    def get_risk_level_indo(self, waste_pct):
        """Kategorisasi level risiko dalam Bahasa Indonesia"""
        if waste_pct < 0.10:
            return {'level': 'RENDAH', 'color': 'success', 'icon': '✓'}
        elif waste_pct < 0.20:
            return {'level': 'SEDANG', 'color': 'warning', 'icon': '⚠'}
        elif waste_pct < 0.35:
            return {'level': 'TINGGI', 'color': 'danger', 'icon': '⚠'}
        else:
            return {'level': 'KRITIS', 'color': 'critical', 'icon': '✗'}
    
    def get_event_name_indo(self, event_type):
        """Konversi nama event ke Bahasa Indonesia"""
        event_names = {
            'Normal': 'Hari Normal',
            'Mudik_Lebaran': 'Mudik Lebaran',
            'Pasca_Mudik': 'Pasca Mudik',
            'Ramadan': 'Bulan Ramadan',
            'Natal_Tahun_Baru': 'Natal & Tahun Baru',
            'Long_Weekend': 'Long Weekend',
            'Hari_Raya_Idul_Adha': 'Hari Raya Idul Adha',
            'Back_to_School': 'Musim Kembali ke Sekolah',
            'Promo_Besar': 'Periode Promo Besar'
        }
        return event_names.get(event_type, event_type)
    
    def get_category_name_indo(self, category):
        """Konversi nama kategori produk ke Bahasa Indonesia"""
        category_names = {
            'Dairy': 'Produk Susu',
            'Bakery': 'Roti & Kue',
            'Meat': 'Daging',
            'Seafood': 'Makanan Laut',
            'Fruits': 'Buah-buahan',
            'Vegetables': 'Sayuran',
            'Beverages': 'Minuman',
            'Snacks': 'Makanan Ringan',
            'Frozen': 'Makanan Beku',
            'Ready_to_Eat': 'Siap Saji'
        }
        return category_names.get(category, category)
    
    def generate_recommendations_indo(self, input_data, waste_pct):
        """Generate rekomendasi dalam Bahasa Indonesia"""
        recommendations = []
        
        # Rekomendasi berdasarkan event
        high_waste_events = ['Mudik_Lebaran', 'Long_Weekend', 'Hari_Raya_Idul_Adha']
        if input_data['event_type'] in high_waste_events:
            event_name = self.get_event_name_indo(input_data['event_type'])
            recommendations.append({
                'prioritas': 'TINGGI',
                'kategori': 'Manajemen Stok',
                'pesan': f'Kurangi pembelian stok 30-50% selama {event_name} karena penurunan jumlah pelanggan.',
                'aksi': 'Hubungi supplier untuk mengurangi order'
            })
        
        # Rekomendasi shelf life pendek
        if input_data['shelf_life_days'] < 5:
            recommendations.append({
                'prioritas': 'TINGGI',
                'kategori': 'Umur Simpan',
                'pesan': 'Produk memiliki umur simpan sangat pendek. Lakukan pemesanan lebih sering dengan kuantitas lebih kecil.',
                'aksi': 'Atur jadwal pengiriman lebih sering (harian/2 hari sekali)'
            })
        
        # Rekomendasi suhu
        if input_data['temperature_deviation'] > 2:
            recommendations.append({
                'prioritas': 'SEDANG',
                'kategori': 'Penyimpanan',
                'pesan': 'Deviasi suhu penyimpanan tinggi dapat mempercepat kerusakan produk.',
                'aksi': 'Periksa dan kalibrasi sistem pendingin, cek seal pintu kulkas'
            })
        
        # Rekomendasi promosi
        if waste_pct > 0.15 and input_data['promotion_active'] == 0:
            recommendations.append({
                'prioritas': 'SEDANG',
                'kategori': 'Pemasaran',
                'pesan': 'Pertimbangkan memberikan diskon atau promo bundle untuk mempercepat perputaran stok.',
                'aksi': 'Buat promo "Beli 2 Gratis 1" atau diskon 20-30%'
            })
        
        # Rekomendasi supplier
        if input_data['supplier_reliability'] < 0.8:
            recommendations.append({
                'prioritas': 'RENDAH',
                'kategori': 'Rantai Pasokan',
                'pesan': 'Tingkat reliabilitas supplier rendah dapat mempengaruhi kualitas dan kesegaran produk.',
                'aksi': 'Evaluasi performa supplier, pertimbangkan cari alternatif'
            })
        
        # Rekomendasi waste tinggi
        if waste_pct > 0.25:
            recommendations.append({
                'prioritas': 'KRITIS',
                'kategori': 'Peringatan Urgent',
                'pesan': 'Prediksi waste sangat tinggi! Segera lakukan review strategi pembelian stok.',
                'aksi': 'Meeting darurat dengan tim purchasing, evaluasi demand forecast'
            })
        
        # Rekomendasi days before expiry
        if input_data['days_before_expiry'] < 3:
            recommendations.append({
                'prioritas': 'TINGGI',
                'kategori': 'Kadaluarsa Dekat',
                'pesan': f"Produk akan kadaluarsa dalam {input_data['days_before_expiry']} hari. Prioritaskan penjualan segera.",
                'aksi': 'Pindahkan ke rak depan, beri label "SEGERA HABISKAN", tawarkan diskon'
            })
        
        return recommendations
    
    def calculate_optimal_stock(self, input_data, waste_pct):
        """Hitung stok optimal untuk minimisasi waste"""
        expected_total_sales = input_data['expected_daily_sales'] * input_data['days_before_expiry']
        
        event_safety_factors = {
            'Normal': 1.15,
            'Mudik_Lebaran': 0.60,
            'Pasca_Mudik': 1.30,
            'Ramadan': 1.20,
            'Natal_Tahun_Baru': 1.25,
            'Long_Weekend': 0.75,
            'Hari_Raya_Idul_Adha': 0.70,
            'Back_to_School': 1.15,
            'Promo_Besar': 1.40
        }
        
        safety_factor = event_safety_factors.get(input_data['event_type'], 1.0)
        waste_adjustment = 1 - (waste_pct * 0.5)
        optimal = int(expected_total_sales * safety_factor * waste_adjustment)
        
        selisih = input_data['initial_stock'] - optimal
        potensi_hemat = max(0, selisih * input_data['price_per_unit'] * waste_pct)
        
        return {
            'stok_disarankan': optimal,
            'stok_saat_ini': input_data['initial_stock'],
            'selisih': selisih,
            'aksi': 'Kurangi' if selisih > 0 else 'Tambah',
            'potensi_penghematan': potensi_hemat
        }
    
    def predict_single(self, input_data):
        """Prediksi untuk satu data input"""
        if not self.is_loaded:
            return None
        
        try:
            # Encode categorical
            product_cat_encoded = self.label_encoders['product_category'].transform([input_data['product_category']])[0]
            event_type_encoded = self.label_encoders['event_type'].transform([input_data['event_type']])[0]
            store_loc_encoded = self.label_encoders['store_location'].transform([input_data['store_location']])[0]
            storage_cond_encoded = self.label_encoders['storage_condition'].transform([input_data['storage_condition']])[0]
            
            features = np.array([[
                product_cat_encoded,
                event_type_encoded,
                store_loc_encoded,
                storage_cond_encoded,
                input_data['shelf_life_days'],
                input_data['days_before_expiry'],
                input_data['initial_stock'],
                input_data['expected_daily_sales'],
                input_data['price_per_unit'],
                input_data['supplier_reliability'],
                input_data['temperature_deviation'],
                input_data['historical_waste_rate'],
                input_data['promotion_active'],
                input_data['day_of_week'],
                input_data['month'],
                input_data['weather_score']
            ]])
            
            features_scaled = self.scaler.transform(features)
            waste_pct = float(self.model.predict(features_scaled, verbose=0)[0][0])
            waste_pct = max(0, min(1, waste_pct))
            
            waste_units = int(input_data['initial_stock'] * waste_pct)
            kerugian = waste_units * input_data['price_per_unit']
            
            return {
                'produk': self.get_category_name_indo(input_data['product_category']),
                'event': self.get_event_name_indo(input_data['event_type']),
                'prediksi_waste_persen': round(waste_pct * 100, 2),
                'prediksi_waste_unit': waste_units,
                'prediksi_kerugian': kerugian,
                'level_risiko': self.get_risk_level_indo(waste_pct),
                'rekomendasi_stok': self.calculate_optimal_stock(input_data, waste_pct),
                'rekomendasi_aksi': self.generate_recommendations_indo(input_data, waste_pct),
                'raw_data': input_data
            }
        except Exception as e:
            return {'error': str(e)}
    
    def predict_from_csv(self, csv_path):
        """Prediksi batch dari file CSV"""
        if not self.is_loaded:
            return {'error': 'Model belum dimuat. Jalankan training terlebih dahulu.'}
        
        try:
            df = pd.read_csv(csv_path)
            results = []
            total_kerugian = 0
            total_potensi_hemat = 0
            
            required_cols = [
                'product_category', 'event_type', 'store_location', 'storage_condition',
                'shelf_life_days', 'days_before_expiry', 'initial_stock', 'expected_daily_sales',
                'price_per_unit', 'supplier_reliability', 'temperature_deviation',
                'historical_waste_rate', 'promotion_active', 'day_of_week', 'month', 'weather_score'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {'error': f'Kolom tidak ditemukan: {", ".join(missing_cols)}'}
            
            for idx, row in df.iterrows():
                input_data = row.to_dict()
                result = self.predict_single(input_data)
                if result and 'error' not in result:
                    result['nomor'] = idx + 1
                    results.append(result)
                    total_kerugian += result['prediksi_kerugian']
                    total_potensi_hemat += result['rekomendasi_stok']['potensi_penghematan']
            
            # Ringkasan
            risk_counts = {'RENDAH': 0, 'SEDANG': 0, 'TINGGI': 0, 'KRITIS': 0}
            for r in results:
                risk_counts[r['level_risiko']['level']] += 1
            
            return {
                'success': True,
                'total_data': len(results),
                'hasil_prediksi': results,
                'ringkasan': {
                    'total_prediksi_kerugian': total_kerugian,
                    'total_potensi_penghematan': total_potensi_hemat,
                    'distribusi_risiko': risk_counts,
                    'rata_rata_waste': round(sum(r['prediksi_waste_persen'] for r in results) / len(results), 2) if results else 0
                }
            }
        except Exception as e:
            return {'error': f'Gagal memproses CSV: {str(e)}'}


# Initialize service
prediction_service = FoodWastePredictionService()


@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', model_loaded=prediction_service.is_loaded)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle upload CSV dan prediksi"""
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
    
    # Prediksi
    results = prediction_service.predict_from_csv(filepath)
    
    # Hapus file setelah diproses
    os.remove(filepath)
    
    if 'error' in results:
        return jsonify(results), 400
    
    return jsonify(results)


@app.route('/template')
def download_template():
    """Provide template info"""
    template_info = {
        'kolom_wajib': [
            {'nama': 'product_category', 'tipe': 'text', 'contoh': 'Dairy', 'pilihan': 'Dairy, Bakery, Meat, Seafood, Fruits, Vegetables, Beverages, Snacks, Frozen, Ready_to_Eat'},
            {'nama': 'event_type', 'tipe': 'text', 'contoh': 'Mudik_Lebaran', 'pilihan': 'Normal, Mudik_Lebaran, Pasca_Mudik, Ramadan, Natal_Tahun_Baru, Long_Weekend, Hari_Raya_Idul_Adha, Back_to_School, Promo_Besar'},
            {'nama': 'store_location', 'tipe': 'text', 'contoh': 'Urban_Center', 'pilihan': 'Urban_Center, Suburban, Rural, Near_Transport_Hub, Residential_Area'},
            {'nama': 'storage_condition', 'tipe': 'text', 'contoh': 'Refrigerated', 'pilihan': 'Refrigerated, Frozen, Room_Temperature'},
            {'nama': 'shelf_life_days', 'tipe': 'number', 'contoh': '7', 'pilihan': 'Jumlah hari (1-365)'},
            {'nama': 'days_before_expiry', 'tipe': 'number', 'contoh': '5', 'pilihan': 'Jumlah hari tersisa'},
            {'nama': 'initial_stock', 'tipe': 'number', 'contoh': '500', 'pilihan': 'Jumlah unit stok'},
            {'nama': 'expected_daily_sales', 'tipe': 'number', 'contoh': '80', 'pilihan': 'Perkiraan penjualan harian'},
            {'nama': 'price_per_unit', 'tipe': 'number', 'contoh': '15000', 'pilihan': 'Harga per unit (Rupiah)'},
            {'nama': 'supplier_reliability', 'tipe': 'decimal', 'contoh': '0.85', 'pilihan': '0.0 - 1.0'},
            {'nama': 'temperature_deviation', 'tipe': 'decimal', 'contoh': '1.5', 'pilihan': 'Deviasi suhu dalam °C'},
            {'nama': 'historical_waste_rate', 'tipe': 'decimal', 'contoh': '0.12', 'pilihan': '0.0 - 1.0'},
            {'nama': 'promotion_active', 'tipe': 'number', 'contoh': '0', 'pilihan': '0 (tidak) atau 1 (ya)'},
            {'nama': 'day_of_week', 'tipe': 'number', 'contoh': '5', 'pilihan': '1 (Senin) - 7 (Minggu)'},
            {'nama': 'month', 'tipe': 'number', 'contoh': '4', 'pilihan': '1 (Januari) - 12 (Desember)'},
            {'nama': 'weather_score', 'tipe': 'number', 'contoh': '3', 'pilihan': '1 (cerah) - 5 (buruk)'}
        ]
    }
    return jsonify(template_info)


if __name__ == '__main__':
    print("=" * 60)
    print("FOOD WASTE PREDICTION - MVP Website")
    print("=" * 60)
    print(f"Model status: {'Loaded' if prediction_service.is_loaded else 'Not loaded'}")
    print("Buka browser: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
