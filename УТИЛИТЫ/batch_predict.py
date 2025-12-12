"""
Validation Script Against Experimental Data
Compare predictions with real test results
"""

import requests
import pandas as pd
import numpy as np

API_URL = "http://localhost:5000"

# ВАШИ ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
experimental_data = [
    {
        "config": {
            "fiber": "E-Glass",
            "matrix": "Polyester",
            "vf": 0.60,
            "layup": "Quasi-isotropic [0/45/90/-45]",
            "manufacturing": "Compression Molding"
        },
        "measured": {
            "tensile_strength": 227.8,
            "tensile_modulus": 14.3,
            "compressive_strength": 149.7,
            "flexural_strength": 292.9,
            "flexural_modulus": 13.0,
            "ilss": 20.8,
            "impact_energy": 14.3
        }
    },
    # Добавьте свои данные здесь
]

def validate_predictions():
    """Validate model against experimental data"""
    
    print("\n" + "="*60)
    print("🔬 VALIDATION AGAINST EXPERIMENTAL DATA")
    print("="*60)
    
    results = []
    
    for i, sample in enumerate(experimental_data):
        print(f"\nSample {i+1}/{len(experimental_data)}:")
        print(f"  Config: {sample['config']['fiber']} / {sample['config']['matrix']}")
        
        response = requests.post(f"{API_URL}/predict", json=sample['config'])
        data = response.json()
        
        if not data['success']:
            print(f"  ❌ Prediction failed: {data['error']}")
            continue
        
        for prop in sample['measured'].keys():
            measured = sample['measured'][prop]
            predicted = data['predictions'][prop]
            uncertainty = data['uncertainty'][prop]
            
            error = abs(predicted - measured)
            error_pct = (error / measured) * 100
            
            in_ci = uncertainty['lower'] <= measured <= uncertainty['upper']
            
            results.append({
                'sample': f"Sample {i+1}",
                'property': prop,
                'measured': measured,
                'predicted': predicted,
                'error_pct': error_pct,
                'in_ci': in_ci
            })
            
            status = "✅" if in_ci else "⚠️"
            print(f"  {status} {prop}: Measured={measured:.1f}, Predicted={predicted:.1f}, Error={error_pct:.1f}%")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📊 OVERALL STATISTICS")
    print("="*60)
    
    print(f"\nTotal predictions: {len(df)}")
    print(f"Within 95% CI: {df['in_ci'].sum()} ({df['in_ci'].mean()*100:.1f}%)")
    print(f"\nMean Error: {df['error_pct'].mean():.1f}%")
    
    df.to_csv('validation_results.csv', index=False)
    print(f"\n✅ Results saved to validation_results.csv")
    
    return df

if __name__ == "__main__":
    validate_predictions()