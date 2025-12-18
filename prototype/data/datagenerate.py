"""
Script untuk generate sample data food waste ke CSV
Jalankan script ini pertama untuk membuat dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_food_waste_data(n_samples=1000):
    """Generate realistic food waste data for retail stores"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Definisi kategori dan karakteristiknya
    product_categories = {
        'Dairy': {'shelf_life_range': (3, 14), 'base_waste': 0.15, 'price_range': (5000, 50000)},
        'Bakery': {'shelf_life_range': (1, 5), 'base_waste': 0.20, 'price_range': (3000, 30000)},
        'Meat': {'shelf_life_range': (2, 7), 'base_waste': 0.18, 'price_range': (20000, 150000)},
        'Seafood': {'shelf_life_range': (1, 4), 'base_waste': 0.22, 'price_range': (25000, 200000)},
        'Fruits': {'shelf_life_range': (3, 10), 'base_waste': 0.12, 'price_range': (5000, 80000)},
        'Vegetables': {'shelf_life_range': (2, 7), 'base_waste': 0.14, 'price_range': (3000, 50000)},
        'Frozen': {'shelf_life_range': (30, 180), 'base_waste': 0.05, 'price_range': (15000, 100000)},
        'Beverages': {'shelf_life_range': (30, 365), 'base_waste': 0.03, 'price_range': (5000, 50000)},
        'Snacks': {'shelf_life_range': (60, 180), 'base_waste': 0.04, 'price_range': (5000, 40000)},
        'Ready_to_Eat': {'shelf_life_range': (1, 3), 'base_waste': 0.25, 'price_range': (10000, 75000)}
    }
    
    # Event types dengan dampak ke penjualan dan waste
    event_types = {
        'Normal': {'sales_multiplier': 1.0, 'waste_multiplier': 1.0, 'weight': 0.40},
        'Mudik_Lebaran': {'sales_multiplier': 0.4, 'waste_multiplier': 2.5, 'weight': 0.10},
        'Pasca_Mudik': {'sales_multiplier': 1.8, 'waste_multiplier': 0.7, 'weight': 0.08},
        'Ramadan': {'sales_multiplier': 1.5, 'waste_multiplier': 1.2, 'weight': 0.10},
        'Natal_Tahun_Baru': {'sales_multiplier': 1.6, 'waste_multiplier': 1.3, 'weight': 0.08},
        'Long_Weekend': {'sales_multiplier': 0.7, 'waste_multiplier': 1.6, 'weight': 0.08},
        'Hari_Raya_Idul_Adha': {'sales_multiplier': 0.6, 'waste_multiplier': 1.8, 'weight': 0.06},
        'Back_to_School': {'sales_multiplier': 1.3, 'waste_multiplier': 0.9, 'weight': 0.05},
        'Promo_Besar': {'sales_multiplier': 2.0, 'waste_multiplier': 0.6, 'weight': 0.05}
    }
    
    # Store locations
    store_locations = ['Urban_Center', 'Suburban', 'Rural', 'Near_Transport_Hub', 'Residential_Area']
    
    # Storage conditions
    storage_conditions = ['Refrigerated', 'Room_Temperature', 'Frozen', 'Controlled_Atmosphere']
    
    data = []
    
    for i in range(n_samples):
        # Random selections
        category = random.choice(list(product_categories.keys()))
        cat_info = product_categories[category]
        
        # Weighted random event selection
        event_names = list(event_types.keys())
        event_weights = [event_types[e]['weight'] for e in event_names]
        event = random.choices(event_names, weights=event_weights, k=1)[0]
        event_info = event_types[event]
        
        location = random.choice(store_locations)
        storage = random.choice(storage_conditions)
        
        # Generate features
        shelf_life = random.randint(*cat_info['shelf_life_range'])
        
        # Days before expiry when product arrived
        days_before_expiry = random.randint(1, max(1, shelf_life - 1))
        
        # Base expected sales (units per day)
        base_daily_sales = random.randint(10, 100)
        expected_sales = int(base_daily_sales * event_info['sales_multiplier'] * random.uniform(0.8, 1.2))
        
        # Initial stock (sometimes over-stocked)
        overstock_factor = random.uniform(1.0, 1.8) if event in ['Mudik_Lebaran', 'Long_Weekend', 'Hari_Raya_Idul_Adha'] else random.uniform(0.9, 1.3)
        initial_stock = int(expected_sales * days_before_expiry * overstock_factor)
        
        # Price per unit
        price_per_unit = random.randint(*cat_info['price_range'])
        
        # Supplier reliability (0-1)
        supplier_reliability = round(random.uniform(0.6, 1.0), 2)
        
        # Temperature deviation (higher = worse storage)
        temp_deviation = round(random.uniform(0, 5) if storage != 'Controlled_Atmosphere' else random.uniform(0, 1), 2)
        
        # Historical waste rate for this store-category combination
        historical_waste_rate = round(cat_info['base_waste'] * random.uniform(0.7, 1.3), 3)
        
        # Promotion active (0 or 1)
        promotion_active = 1 if random.random() < 0.3 else 0
        
        # Day of week (1-7, 1=Monday)
        day_of_week = random.randint(1, 7)
        
        # Month (1-12)
        month = random.randint(1, 12)
        
        # Weather condition score (1-5, 5=bad weather)
        weather_score = random.randint(1, 5)
        
        # Calculate actual sales (affected by various factors)
        location_factor = {'Urban_Center': 1.2, 'Suburban': 1.0, 'Rural': 0.7, 
                          'Near_Transport_Hub': 1.1, 'Residential_Area': 0.9}[location]
        
        weather_factor = 1 - (weather_score - 1) * 0.05
        promo_factor = 1.3 if promotion_active else 1.0
        weekend_factor = 1.2 if day_of_week in [6, 7] else 1.0
        
        actual_daily_sales = int(expected_sales * location_factor * weather_factor * promo_factor * weekend_factor * random.uniform(0.7, 1.1))
        total_actual_sales = min(initial_stock, actual_daily_sales * days_before_expiry)
        
        # Calculate waste
        remaining_stock = max(0, initial_stock - total_actual_sales)
        
        # Waste percentage calculation with multiple factors
        base_waste_pct = cat_info['base_waste'] * event_info['waste_multiplier']
        temp_impact = temp_deviation * 0.02
        reliability_impact = (1 - supplier_reliability) * 0.1
        shelf_life_impact = max(0, (7 - shelf_life) * 0.02) if shelf_life < 7 else 0
        
        waste_percentage = base_waste_pct + temp_impact + reliability_impact + shelf_life_impact
        waste_percentage = waste_percentage * random.uniform(0.8, 1.2)
        waste_percentage = round(min(0.95, max(0.01, waste_percentage)), 4)
        
        # Actual waste units
        waste_units = int(remaining_stock * waste_percentage) + random.randint(0, 5)
        waste_units = min(waste_units, initial_stock)
        
        # Actual waste percentage (recalculated)
        actual_waste_pct = round(waste_units / initial_stock if initial_stock > 0 else 0, 4)
        
        # Financial loss
        financial_loss = waste_units * price_per_unit
        
        data.append({
            'record_id': i + 1,
            'product_category': category,
            'event_type': event,
            'store_location': location,
            'storage_condition': storage,
            'shelf_life_days': shelf_life,
            'days_before_expiry': days_before_expiry,
            'initial_stock': initial_stock,
            'expected_daily_sales': expected_sales,
            'actual_daily_sales': actual_daily_sales,
            'total_actual_sales': total_actual_sales,
            'price_per_unit': price_per_unit,
            'supplier_reliability': supplier_reliability,
            'temperature_deviation': temp_deviation,
            'historical_waste_rate': historical_waste_rate,
            'promotion_active': promotion_active,
            'day_of_week': day_of_week,
            'month': month,
            'weather_score': weather_score,
            'waste_units': waste_units,
            'waste_percentage': actual_waste_pct,
            'financial_loss': financial_loss
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING FOOD WASTE SAMPLE DATA")
    print("=" * 60)
    
    # Generate data
    df = generate_food_waste_data(500)
    
    # Save to CSV
    csv_path = "food_waste_dataset2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Dataset saved to: {csv_path}")
    print(f"Total records: {len(df)}")
    
    # Display statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print("\n--- Waste by Event Type ---")
    event_stats = df.groupby('event_type').agg({
        'waste_percentage': 'mean',
        'financial_loss': 'sum',
        'record_id': 'count'
    }).round(4)
    event_stats.columns = ['Avg Waste %', 'Total Loss (Rp)', 'Count']
    print(event_stats.sort_values('Avg Waste %', ascending=False))
    
    print("\n--- Waste by Product Category ---")
    cat_stats = df.groupby('product_category').agg({
        'waste_percentage': 'mean',
        'financial_loss': 'sum',
        'record_id': 'count'
    }).round(4)
    cat_stats.columns = ['Avg Waste %', 'Total Loss (Rp)', 'Count']
    print(cat_stats.sort_values('Avg Waste %', ascending=False))
    
    print("\n--- Sample Data Preview ---")
    print(df.head(10).to_string())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
