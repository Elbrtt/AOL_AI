"""
Backend untuk inference/prediction food waste
Load model terlatih dan gunakan untuk prediksi
"""

import numpy as np
import pickle
from tensorflow import keras


class FoodWastePredictionService:
    """Service class untuk prediksi food waste"""
    
    def __init__(self, model_path='food_waste_model.keras', artifacts_path='model_artifacts.pkl'):
        """Initialize service dengan load model dan artifacts"""
        
        print("[INFO] Loading model and artifacts...")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"[INFO] Model loaded from: {model_path}")
        
        # Load artifacts (scaler, encoders)
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.feature_cols = artifacts['feature_cols']
        
        print("[INFO] Artifacts loaded successfully")
        print("[INFO] Service ready for predictions")
    
    def get_valid_categories(self):
        """Return valid categories for each categorical field"""
        return {
            col: list(encoder.classes_) 
            for col, encoder in self.label_encoders.items()
        }
    
    def prepare_features(self, input_data):
        """Prepare input features for prediction"""
        
        # Encode categorical variables
        product_cat_encoded = self.label_encoders['product_category'].transform([input_data['product_category']])[0]
        event_type_encoded = self.label_encoders['event_type'].transform([input_data['event_type']])[0]
        store_loc_encoded = self.label_encoders['store_location'].transform([input_data['store_location']])[0]
        storage_cond_encoded = self.label_encoders['storage_condition'].transform([input_data['storage_condition']])[0]
        
        # Create feature array
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
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, input_data):
        """
        Predict waste percentage for given input
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing all required features
            
        Returns:
        --------
        dict : Prediction results with recommendations
        """
        
        # Prepare features
        features = self.prepare_features(input_data)
        
        # Predict
        waste_percentage = float(self.model.predict(features, verbose=0)[0][0])
        waste_percentage = max(0, min(1, waste_percentage))  # Clamp between 0-1
        
        # Calculate waste units and financial loss
        waste_units = int(input_data['initial_stock'] * waste_percentage)
        financial_loss = waste_units * input_data['price_per_unit']
        
        # Generate recommendations
        recommendations = self._generate_recommendations(input_data, waste_percentage)
        
        # Calculate optimal stock
        optimal_stock = self._calculate_optimal_stock(input_data, waste_percentage)
        
        return {
            'predicted_waste_percentage': round(waste_percentage * 100, 2),
            'predicted_waste_units': waste_units,
            'predicted_financial_loss': financial_loss,
            'optimal_stock_recommendation': optimal_stock,
            'risk_level': self._get_risk_level(waste_percentage),
            'recommendations': recommendations
        }
    
    def _get_risk_level(self, waste_pct):
        """Categorize risk level based on waste percentage"""
        if waste_pct < 0.10:
            return 'LOW'
        elif waste_pct < 0.20:
            return 'MEDIUM'
        elif waste_pct < 0.35:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _calculate_optimal_stock(self, input_data, waste_pct):
        """Calculate optimal stock to minimize waste"""
        
        expected_total_sales = input_data['expected_daily_sales'] * input_data['days_before_expiry']
        
        # Safety factor based on event type
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
        
        # Adjust based on predicted waste
        waste_adjustment = 1 - (waste_pct * 0.5)
        
        optimal = int(expected_total_sales * safety_factor * waste_adjustment)
        
        return {
            'recommended_stock': optimal,
            'vs_current': input_data['initial_stock'] - optimal,
            'potential_savings': max(0, (input_data['initial_stock'] - optimal) * input_data['price_per_unit'] * waste_pct)
        }
    
    def _generate_recommendations(self, input_data, waste_pct):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Event-based recommendations
        high_waste_events = ['Mudik_Lebaran', 'Long_Weekend', 'Hari_Raya_Idul_Adha']
        if input_data['event_type'] in high_waste_events:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'STOCK_MANAGEMENT',
                'message': f"Kurangi stok 30-50% saat {input_data['event_type'].replace('_', ' ')} karena penurunan traffic pelanggan"
            })
        
        # Shelf life recommendations
        if input_data['shelf_life_days'] < 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'SHELF_LIFE',
                'message': "Produk dengan shelf life pendek - pertimbangkan stok lebih sering dengan quantity lebih kecil"
            })
        
        # Temperature recommendations
        if input_data['temperature_deviation'] > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'STORAGE',
                'message': "Periksa sistem pendingin - deviasi suhu tinggi meningkatkan waste"
            })
        
        # Promotion recommendations
        if waste_pct > 0.15 and input_data['promotion_active'] == 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'MARKETING',
                'message': "Pertimbangkan promo/diskon untuk mempercepat perputaran stok"
            })
        
        # Supplier recommendations
        if input_data['supplier_reliability'] < 0.8:
            recommendations.append({
                'priority': 'LOW',
                'category': 'SUPPLY_CHAIN',
                'message': "Evaluasi performa supplier - reliability rendah mempengaruhi kualitas stok"
            })
        
        # General high waste recommendation
        if waste_pct > 0.25:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'URGENT',
                'message': "Waste diprediksi sangat tinggi - segera review strategi stocking"
            })
        
        return recommendations
    
    def batch_predict(self, input_data_list):
        """Predict for multiple inputs"""
        return [self.predict(data) for data in input_data_list]


# Demo function
def run_demo():
    """Run demo predictions"""
    
    print("=" * 70)
    print("FOOD WASTE PREDICTION SERVICE - DEMO")
    print("=" * 70)
    
    # Initialize service
    service = FoodWastePredictionService()
    
    # Print valid categories
    print("\n--- Valid Input Categories ---")
    for col, values in service.get_valid_categories().items():
        print(f"{col}: {values}")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Scenario 1: Dairy product during Mudik Lebaran',
            'data': {
                'product_category': 'Dairy',
                'event_type': 'Mudik_Lebaran',
                'store_location': 'Urban_Center',
                'storage_condition': 'Refrigerated',
                'shelf_life_days': 7,
                'days_before_expiry': 5,
                'initial_stock': 500,
                'expected_daily_sales': 80,
                'price_per_unit': 15000,
                'supplier_reliability': 0.85,
                'temperature_deviation': 1.5,
                'historical_waste_rate': 0.12,
                'promotion_active': 0,
                'day_of_week': 5,
                'month': 4,
                'weather_score': 3
            }
        },
        {
            'name': 'Scenario 2: Bakery on Normal Day',
            'data': {
                'product_category': 'Bakery',
                'event_type': 'Normal',
                'store_location': 'Suburban',
                'storage_condition': 'Room_Temperature',
                'shelf_life_days': 3,
                'days_before_expiry': 2,
                'initial_stock': 200,
                'expected_daily_sales': 90,
                'price_per_unit': 12000,
                'supplier_reliability': 0.90,
                'temperature_deviation': 0.5,
                'historical_waste_rate': 0.18,
                'promotion_active': 1,
                'day_of_week': 6,
                'month': 8,
                'weather_score': 2
            }
        },
        {
            'name': 'Scenario 3: Frozen food during Ramadan',
            'data': {
                'product_category': 'Frozen',
                'event_type': 'Ramadan',
                'store_location': 'Residential_Area',
                'storage_condition': 'Frozen',
                'shelf_life_days': 90,
                'days_before_expiry': 60,
                'initial_stock': 300,
                'expected_daily_sales': 25,
                'price_per_unit': 45000,
                'supplier_reliability': 0.95,
                'temperature_deviation': 0.2,
                'historical_waste_rate': 0.05,
                'promotion_active': 0,
                'day_of_week': 3,
                'month': 3,
                'weather_score': 2
            }
        },
        {
            'name': 'Scenario 4: Seafood during Long Weekend',
            'data': {
                'product_category': 'Seafood',
                'event_type': 'Long_Weekend',
                'store_location': 'Near_Transport_Hub',
                'storage_condition': 'Refrigerated',
                'shelf_life_days': 3,
                'days_before_expiry': 2,
                'initial_stock': 150,
                'expected_daily_sales': 40,
                'price_per_unit': 75000,
                'supplier_reliability': 0.75,
                'temperature_deviation': 2.5,
                'historical_waste_rate': 0.20,
                'promotion_active': 0,
                'day_of_week': 7,
                'month': 5,
                'weather_score': 4
            }
        }
    ]
    
    # Run predictions
    for scenario in test_scenarios:
        print("\n" + "=" * 70)
        print(f">>> {scenario['name']}")
        print("=" * 70)
        
        result = service.predict(scenario['data'])
        
        print(f"\n  PREDICTION RESULTS:")
        print(f"  -------------------")
        print(f"  Predicted Waste: {result['predicted_waste_percentage']}%")
        print(f"  Waste Units: {result['predicted_waste_units']} units")
        print(f"  Financial Loss: Rp {result['predicted_financial_loss']:,}")
        print(f"  Risk Level: {result['risk_level']}")
        
        print(f"\n  STOCK RECOMMENDATION:")
        print(f"  ---------------------")
        opt = result['optimal_stock_recommendation']
        print(f"  Current Stock: {scenario['data']['initial_stock']} units")
        print(f"  Recommended Stock: {opt['recommended_stock']} units")
        print(f"  Reduce by: {opt['vs_current']} units")
        print(f"  Potential Savings: Rp {opt['potential_savings']:,.0f}")
        
        print(f"\n  RECOMMENDATIONS:")
        print(f"  ----------------")
        for rec in result['recommendations']:
            print(f"  [{rec['priority']}] {rec['category']}: {rec['message']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
