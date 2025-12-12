"""
Hybrid PIRF Training Pipeline
Trains Random Forest models for composite property prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Material properties
FIBERS = {
    'Carbon T300': {'E': 230, 'sigma': 3530, 'rho': 1.76},
    'E-Glass': {'E': 73, 'sigma': 3450, 'rho': 2.54},
    'Kevlar 49': {'E': 131, 'sigma': 3620, 'rho': 1.44},
    'Basalt': {'E': 89, 'sigma': 4840, 'rho': 2.75},
    'Flax': {'E': 58, 'sigma': 1100, 'rho': 1.50}
}

MATRICES = {
    'Epoxy': {'E': 3.2, 'sigma': 78, 'rho': 1.20},
    'Polyester': {'E': 3.5, 'sigma': 55, 'rho': 1.15},
    'Vinyl Ester': {'E': 3.4, 'sigma': 82, 'rho': 1.14},
    'PEEK': {'E': 3.9, 'sigma': 105, 'rho': 1.32},
    'Polyamide 6': {'E': 2.8, 'sigma': 82, 'rho': 1.14}
}

LAYUPS = [
    'Unidirectional 0°', 'Unidirectional 90°', 'Woven 0/90',
    'Quasi-isotropic [0/45/90/-45]', 'Angle-ply [±45]',
    'Cross-ply [0/90]', 'Random Mat'
]

MANUFACTURING = [
    'Autoclave', 'VARTM', 'RTM', 'Compression Molding',
    'Hand Layup', 'Filament Winding', 'Pultrusion'
]

print("="*60)
print("HYBRID PIRF TRAINING PIPELINE")
print("="*60)

# Generate synthetic training data
def generate_training_data(n_samples=363):
    """Generate physics-informed synthetic data"""
    
    print("⚠ No database found. Generating synthetic training data...")
    
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        fiber_name = np.random.choice(list(FIBERS.keys()))
        matrix_name = np.random.choice(list(MATRICES.keys()))
        Vf = np.random.uniform(0.40, 0.60)
        layup = np.random.choice(LAYUPS)
        manufacturing = np.random.choice(MANUFACTURING)
        
        fiber = FIBERS[fiber_name]
        matrix = MATRICES[matrix_name]
        
        # Physics-based calculations with noise
        E_L = fiber['E'] * Vf + matrix['E'] * (1 - Vf)
        sigma_uts = fiber['sigma'] * Vf * 0.9 + matrix['sigma'] * (1 - Vf) * 0.45
        
        # Add realistic noise
        noise_factor = np.random.normal(1.0, 0.15)
        
        data.append({
            'fiber': fiber_name,
            'matrix': matrix_name,
            'vf': Vf,
            'layup': layup,
            'manufacturing': manufacturing,
            'tensile_strength': sigma_uts * noise_factor * 0.262,
            'tensile_modulus': E_L * noise_factor * 0.72,
            'compressive_strength': sigma_uts * 0.75 * noise_factor * 0.202,
            'flexural_strength': sigma_uts * 1.25 * noise_factor * 0.289,
            'flexural_modulus': E_L * 1.05 * noise_factor * 0.60,
            'ilss': matrix['sigma'] * 0.50 * (1 - 0.5 * Vf) * noise_factor * 1.49,
            'impact_energy': (fiber['E'] * Vf * 0.015 + matrix['E'] * (1 - Vf) * 0.01) * noise_factor * 0.98
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/training_data.csv', index=False)
    print(f"✓ Generated and saved {len(df)} synthetic records")
    
    return df

# Load or generate data
data_path = 'data/training_data.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records from {data_path}")
else:
    df = generate_training_data()

print(f"\nDataset: {len(df)} samples")
print(f"  Fibers: {df['fiber'].nunique()}")
print(f"  Matrices: {df['matrix'].nunique()}")
print(f"  Vf range: [{df['vf'].min():.2f}, {df['vf'].max():.2f}]")

# Feature engineering
def create_features(df):
    """Create 26 physics-informed features"""
    
    print("\nPreparing physics-informed features...")
    
    X = []
    
    fiber_encoding = {f: i for i, f in enumerate(FIBERS.keys())}
    matrix_encoding = {m: i for i, m in enumerate(MATRICES.keys())}
    layup_encoding = {l: i for i, l in enumerate(LAYUPS)}
    mfg_encoding = {m: i for i, m in enumerate(MANUFACTURING)}
    
    for _, row in df.iterrows():
        fiber = FIBERS[row['fiber']]
        matrix = MATRICES[row['matrix']]
        Vf = row['vf']
        
        # Base features (5)
        features = [
            fiber_encoding[row['fiber']],
            matrix_encoding[row['matrix']],
            Vf,
            layup_encoding[row['layup']],
            mfg_encoding[row['manufacturing']]
        ]
        
        # Physics-derived (7)
        E_L = fiber['E'] * Vf + matrix['E'] * (1 - Vf)
        sigma_uts = fiber['sigma'] * Vf * 0.9
        features.extend([
            sigma_uts * 0.262,
            E_L * 0.72,
            sigma_uts * 0.75 * 0.202,
            sigma_uts * 1.25 * 0.289,
            E_L * 1.05 * 0.60,
            matrix['sigma'] * 0.50 * (1 - 0.5 * Vf) * 1.49,
            (fiber['E'] * Vf * 0.015) * 0.98
        ])
        
        # Ratios (4)
        features.extend([
            fiber['E'] / matrix['E'],
            fiber['sigma'] / matrix['sigma'],
            fiber['rho'] / matrix['rho'],
            1.0  # G_f/G_m approximation
        ])
        
        # Vf transformations (3)
        features.extend([
            Vf ** 2,
            Vf ** 3,
            1 / (1 - Vf + 1e-6)
        ])
        
        # Interactions (7)
        features.extend([
            Vf * (fiber['E'] / matrix['E']),
            Vf * (fiber['sigma'] / matrix['sigma']),
            E_L ** 2,
            sigma_uts * E_L,
            Vf * E_L,
            (1 - Vf) * matrix['E'],
            (fiber['E'] / matrix['E']) * (fiber['sigma'] / matrix['sigma'])
        ])
        
        X.append(features)
    
    X = np.array(X)
    print(f"  Feature matrix: {X.shape}")
    
    return X

# Prepare features and targets
X = create_features(df)

target_properties = [
    'tensile_strength', 'tensile_modulus', 'compressive_strength',
    'flexural_strength', 'flexural_modulus', 'ilss', 'impact_energy'
]

y = df[target_properties].values
print(f"  Targets: {y.shape}")
print(f"  Features per sample: {X.shape[1]}")

# Train models
print("\n" + "="*60)
print("TRAINING HYBRID PIRF MODELS")
print("="*60)

models = {}
results = []

for i, prop in enumerate(target_properties):
    print(f"\n--- Training: {prop} ---")
    
    y_prop = y[:, i]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_prop, test_size=0.2, random_state=42
    )
    
    # Scale features
    if i == 0:  # Only create scaler once
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    r2_train = rf.score(X_train_scaled, y_train)
    r2_test = rf.score(X_test_scaled, y_test)
    
    y_pred = rf.predict(X_test_scaled)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    print(f"  R² (train): {r2_train:.3f}")
    print(f"  R² (test):  {r2_test:.3f}")
    print(f"  MAE (test): {mae:.2f}")
    print(f"  RMSE (test): {rmse:.2f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    models[prop] = {
        'rf': rf,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    results.append({
        'property': prop,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'r2_cv_mean': cv_scores.mean(),
        'r2_cv_std': cv_scores.std(),
        'mae': mae,
        'rmse': rmse
    })

# Save models
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\nAverage R² (test): {results_df['r2_test'].mean():.3f}")
print(f"Average R² (CV):   {results_df['r2_cv_mean'].mean():.3f}")

# Save to disk
with open('models/hybrid_model.pkl', 'wb') as f:
    pickle.dump(models, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

results_df.to_csv('models/training_results.csv', index=False)

print("\n✓ Models saved to models/")
print("  - hybrid_model.pkl")
print("  - scaler.pkl")
print("  - training_results.csv")

print("\n✓ Training complete!")
