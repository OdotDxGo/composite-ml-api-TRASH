"""
Sensitivity Analysis
Analyze how input parameters affect predictions
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:5000"

def get_prediction(config):
    """Get prediction from API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=config)
        data = response.json()
        if data['success']:
            return data['predictions']
        return None
    except:
        return None

def vf_sensitivity_analysis():
    """Analyze sensitivity to volume fraction"""
    
    print("\n" + "="*60)
    print("📊 VOLUME FRACTION SENSITIVITY ANALYSIS")
    print("="*60)
    
    base_config = {
        'fiber': 'E-Glass',
        'matrix': 'Epoxy',
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'VARTM'
    }
    
    vf_values = np.linspace(0.30, 0.70, 21)
    
    results = []
    
    print(f"\nTesting {len(vf_values)} volume fractions...")
    
    for vf in vf_values:
        config = {**base_config, 'vf': float(vf)}
        pred = get_prediction(config)
        
        if pred:
            results.append({
                'vf': vf,
                **pred
            })
    
    df = pd.DataFrame(results)
    
    print("\n📈 Sensitivity Metrics (per 0.1 Vf increase):")
    
    properties = ['tensile_strength', 'tensile_modulus', 'ilss']
    
    for prop in properties:
        coeffs = np.polyfit(df['vf'], df[prop], 1)
        slope = coeffs[0]
        sensitivity = slope * 0.1
        print(f"  {prop}: {sensitivity:+.2f} per 0.1 Vf increase")
    
    # Save
    df.to_csv('vf_sensitivity.csv', index=False)
    print("\n✅ Results saved: vf_sensitivity.csv")
    
    return df

def material_sensitivity_analysis():
    """Analyze sensitivity to material selection"""
    
    print("\n" + "="*60)
    print("🔬 MATERIAL SELECTION SENSITIVITY")
    print("="*60)
    
    fibers = ['E-Glass', 'Carbon T300', 'Kevlar 49', 'Basalt']
    matrices = ['Epoxy', 'Polyester', 'Vinyl Ester']
    
    base_config = {
        'vf': 0.55,
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'VARTM'
    }
    
    results = []
    
    print(f"\nTesting {len(fibers) * len(matrices)} material combinations...")
    
    for fiber in fibers:
        for matrix in matrices:
            config = {**base_config, 'fiber': fiber, 'matrix': matrix}
            pred = get_prediction(config)
            
            if pred:
                results.append({
                    'fiber': fiber,
                    'matrix': matrix,
                    'combination': f"{fiber}/{matrix}",
                    **pred
                })
    
    df = pd.DataFrame(results)
    
    print("\n🏆 Material Rankings (by tensile strength):")
    ranked = df.sort_values('tensile_strength', ascending=False)
    for i, row in ranked.head(5).iterrows():
        print(f"  {row['fiber']} + {row['matrix']}: {row['tensile_strength']:.1f} MPa")
    
    df.to_csv('material_sensitivity.csv', index=False)
    print("\n✅ Results saved: material_sensitivity.csv")
    
    return df

def layup_sensitivity_analysis():
    """Analyze sensitivity to layup configuration"""
    
    print("\n" + "="*60)
    print("🧵 LAYUP CONFIGURATION SENSITIVITY")
    print("="*60)
    
    layups = [
        'Unidirectional 0°',
        'Woven 0/90',
        'Quasi-isotropic [0/45/90/-45]',
        'Cross-ply [0/90]'
    ]
    
    base_config = {
        'fiber': 'E-Glass',
        'matrix': 'Epoxy',
        'vf': 0.55,
        'manufacturing': 'VARTM'
    }
    
    results = []
    
    for layup in layups:
        config = {**base_config, 'layup': layup}
        pred = get_prediction(config)
        
        if pred:
            results.append({
                'layup': layup,
                **pred
            })
    
    df = pd.DataFrame(results)
    
    print("\n📐 Layup Efficiency (normalized by unidirectional 0°):")
    reference = df[df['layup'] == 'Unidirectional 0°']['tensile_strength'].values[0]
    
    for _, row in df.iterrows():
        efficiency = (row['tensile_strength'] / reference) * 100
        print(f"  {row['layup']}: {efficiency:.1f}%")
    
    df.to_csv('layup_sensitivity.csv', index=False)
    print("\n✅ Results saved: layup_sensitivity.csv")
    
    return df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 SENSITIVITY ANALYSIS SUITE")
    print("="*60)
    
    print("\n[1] Volume Fraction Sensitivity")
    print("[2] Material Selection Sensitivity")
    print("[3] Layup Configuration Sensitivity")
    print("[4] Run All Analyses")
    
    choice = input("\nSelect analysis (1-4): ").strip()
    
    if choice == '1':
        vf_sensitivity_analysis()
    elif choice == '2':
        material_sensitivity_analysis()
    elif choice == '3':
        layup_sensitivity_analysis()
    elif choice == '4':
        print("\n🚀 Running all analyses...\n")
        vf_sensitivity_analysis()
        material_sensitivity_analysis()
        layup_sensitivity_analysis()
    else:
        print("Invalid choice. Running volume fraction analysis...")
        vf_sensitivity_analysis()