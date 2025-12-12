"""
Training script for Hybrid PIRF models
Trains Random Forest models for each property using physics-informed features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import physics engine from app
import sys
sys.path.append('.')
from app import PhysicsEngine, FeatureEngineer, FIBERS, MATRICES

def load_database():
    """
    Load composite materials database
    
    Returns:
        pd.DataFrame with columns:
        - fiber, matrix, vf, layup, manufacturing
        - tensile_strength, tensile_modulus, compressive_strength
        - flexural_strength, flexural_modulus, ilss, impact_energy
    """
    
    # Check if database exists
    db_path = 'data/composite_database.csv'
    
    if os.path.exists(db_path):
        df = pd.read_csv(db_path)
        print(f"✓ Loaded {len(df)} records from database")
        return df
    
    # Generate synthetic database based on ROM + noise
    # This simulates real experimental data
    print("⚠ No database found. Generating synthetic training data...")
    
    np.random.seed(42)
    
    data = []
    physics_engine = PhysicsEngine()
    
    # Sample combinations
    fibers = list(FIBERS.keys())
    matrices = list(MATRICES.keys())
    layups = ['Unidirectional 0°', 'Woven 0/90', 'Quasi-isotropic [0/45/90/-45]']
    mfg_processes = ['Autoclave', 'VARTM', 'Compression Molding', 'Hand Layup']
    
    for fiber_name in fibers:
        for matrix_name in matrices:
            for layup in layups:
                for mfg in mfg_processes:
                    # Sample 3 Vf values
                    for Vf in [0.40, 0.50, 0.60]:
                        
                        fiber = FIBERS[fiber_name]
                        matrix = MATRICES[matrix_name]
                        
                        # Calculate ROM properties
                        rom_props = physics_engine.calculate_rom_properties(
                            fiber, matrix, Vf, layup, mfg
                        )
                        
                        # Add realistic noise (simulate experimental variation)
                        noise_level = 0.08  # 8% coefficient of variation
                        
                        record = {
                            'fiber': fiber_name,
                            'matrix': matrix_name,
                            'vf': Vf,
                            'layup': layup,
                            'manufacturing': mfg,
                            'tensile_strength': rom_props['tensile_strength'] * np.random.normal(1.0, noise_level),
                            'tensile_modulus': rom_props['tensile_modulus'] * np.random.normal(1.0, noise_level),
                            'compressive_strength': rom_props['compressive_strength'] * np.random.normal(1.0, noise_level),
                            'flexural_strength': rom_props['flexural_strength'] * np.random.normal(1.0, noise_level),
                            'flexural_modulus': rom_props['flexural_modulus'] * np.random.normal(1.0, noise_level),
                            'ilss': rom_props['ilss'] * np.random.normal(1.0, noise_level),
                            'impact_energy': rom_props['impact_energy'] * np.random.normal(1.0, noise_level)
                        }
                        
                        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Save database
    os.makedirs('data', exist_ok=True)
    df.to_csv(db_path, index=False)
    print(f"✓ Generated and saved {len(df)} synthetic records")
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    
    physics_engine = PhysicsEngine()
    feature_engineer = FeatureEngineer()
    
    X_list = []
    
    for idx, row in df.iterrows():
        fiber = FIBERS[row['fiber']]
        matrix = MATRICES[row['matrix']]
        
        # Calculate ROM properties
        rom_props = physics_engine.calculate_rom_properties(
            fiber, matrix, row['vf'], row['layup'], row['manufacturing']
        )
        
        # Create features
        features = feature_engineer.create_features(
            row['fiber'], row['matrix'], row['vf'], 
            row['layup'], row['manufacturing'], rom_props
        )
        
        X_list.append(features.flatten())
    
    X = np.array(X_list)
    
    # Target variables
    properties = ['tensile_strength', 'tensile_modulus', 'compressive_strength',
                 'flexural_strength', 'flexural_modulus', 'ilss', 'impact_energy']
    
    y = df[properties].values
    
    return X, y, properties

def train_models(X, y, property_names):
    """Train Random Forest models for each property"""
    
    print("\n" + "="*60)
    print("TRAINING HYBRID PIRF MODELS")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = []
    
    for i, prop in enumerate(property_names):
        print(f"\n--- Training: {prop} ---")
        
        y_train_prop = y_train[:, i]
        y_test_prop = y_test[:, i]
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=150,  # Reduced for speed (was 500)
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train_prop)
        
        # Predictions
        y_pred_train = rf.predict(X_train_scaled)
        y_pred_test = rf.predict(X_test_scaled)
        
        # Metrics
        r2_train = r2_score(y_train_prop, y_pred_train)
        r2_test = r2_score(y_test_prop, y_pred_test)
        mae_test = mean_absolute_error(y_test_prop, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_prop, y_pred_test))
        
        print(f"  R² (train): {r2_train:.3f}")
        print(f"  R² (test):  {r2_test:.3f}")
        print(f"  MAE (test): {mae_test:.2f}")
        print(f"  RMSE (test): {rmse_test:.2f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_train_scaled, y_train_prop, 
                                    cv=5, scoring='r2', n_jobs=-1)
        print(f"  CV R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Store model
        models[prop] = {
            'rf': rf,
            'r2_test': r2_test,
            'mae_test': mae_test
        }
        
        results.append({
            'property': prop,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'r2_cv_mean': cv_scores.mean(),
            'r2_cv_std': cv_scores.std(),
            'mae': mae_test,
            'rmse': rmse_test
        })
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\nAverage R² (test): {results_df['r2_test'].mean():.3f}")
    print(f"Average R² (CV):   {results_df['r2_cv_mean'].mean():.3f}")
    
    return models, scaler, results_df

def save_models(models, scaler):
    """Save trained models"""
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/hybrid_model.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✓ Models saved to models/")
    print("  - hybrid_model.pkl")
    print("  - scaler.pkl")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("HYBRID PIRF TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_database()
    print(f"\nDataset: {len(df)} samples")
    print(f"  Fibers: {df['fiber'].nunique()}")
    print(f"  Matrices: {df['matrix'].nunique()}")
    print(f"  Vf range: [{df['vf'].min():.2f}, {df['vf'].max():.2f}]")
    
    # Prepare features
    print("\nPreparing physics-informed features...")
    X, y, property_names = prepare_features(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  Features per sample: {X.shape[1]}")
    
    # Train models
    models, scaler, results = train_models(X, y, property_names)
    
    # Save models
    save_models(models, scaler)
    
    # Save results
    results.to_csv('models/training_results.csv', index=False)
    print("  - training_results.csv")
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("  1. Review training_results.csv")
    print("  2. Test predictions with app.py")
    print("  3. Deploy to Railway")

if __name__ == '__main__':
    main()