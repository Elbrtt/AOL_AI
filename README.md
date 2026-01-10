==== FOOD WASTE ANALYZER (Most Viable Product) ====
- Aurelius Elbert 2802391555
- Brandon Maximillian 2802418512
- Jordhy Alexander 2802389216

Projek ini menggunakan machine learning dalam memprediksi kemungkinan kerusakan pada sebuah produk makanan
yang akan digunakan untuk menghitung potensial berapa banyak unit makanan yang terbuang.

Apa yang di prediksi AI?
- **Spoilage Chance**
  - Kolom yang digunakan dari dataset:
    > category_encode', store_location_encoded, brand_encoded, StockQty, DaysUntilExpiry,
                                          DailySaleAvg, DistanceToNearestStore, AvgDailySaleInNearbyStores, temp_encoded

Fitur yang di hard-coded:
1. Rekomendasi
   - berdasarkan spoilage chance
   - berdasarkan DaysUntilExpiry
3. Kalkulasi
   - potensial waste = unsold * spoilage chance
   - unsold = stock_qty - daily sales * days till expiry
   - waste percentage (%) = potensi waste / stok * 100

PROSEDUR
1. Download Requirements.txt  dan jalankan enviroment
2. Jalankan AppTest.py kemudian click link localHost 5000 dari output program
3. Masukkan file CSV sesuai dengan format yang ditentukan

*Folder Upload digunakan untuk memastikan file yang ingin diupload ke dalam projek untuk dianalisis
