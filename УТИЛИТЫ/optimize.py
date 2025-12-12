"""
Optimization Script
Find optimal composite configuration for given requirements
Uses genetic algorithm for multi-objective optimization
"""

import requests
import numpy as np
from scipy.optimize import differential_evolution

API_URL = "http://localhost:5000"

# Material encoding
FIBER_MAP = {'E-Glass': 0, 'Carbon T300': 1, 'Kevlar 49': 2, 'Basalt': 3, 'Flax': 4}
MATRIX_MAP = {'Epoxy': 0, 'Polyester': 1, 'Vinyl Ester': 2, 'PEEK': 3, 'Polyamide 6': 4}
LAYUP_MAP = {
    'Unidirectional 0°': 0, 'Unidirectional 90°': 1, 'Woven 0/90': 2,
    'Quasi-isotropic [0/45/90/-45]': 3, 'Angle-ply [±45]': 4,
    'Cross-ply [0/90]': 5, 'Random Mat': 6
}
MFG_MAP = {
    'Autoclave': 0, 'VARTM': 1, 'RTM': 2, 'Compression Molding': 3,
    'Hand Layup': 4, 'Filament Winding': 5, 'Pultrusion': 6
}

FIBER_NAMES = list(FIBER_MAP.keys())
MATRIX_NAMES = list(MATRIX_MAP.keys())
LAYUP_NAMES = list(LAYUP_MAP.keys())
MFG_NAMES = list(MFG_MAP.keys())

def decode_configuration(x):
    """Convert optimization variables to configuration"""
    fiber = FIBER_NAMES[int(round(x[0]))]
    matrix = MATRIX_NAMES[int(round(x[1]))]
    vf = x[2]
    layup = LAYUP_NAMES[int(round(x[3]))]
    manufacturing = MFG_NAMES[int(round(x[4]))]
    
    return {
        'fiber': fiber,
        'matrix': matrix,
        'vf': float(vf),
        'layup': layup,
        'manufacturing': manufacturing
    }

def get_prediction(config):
    """Get prediction from API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=config, timeout=5)
        data = response.json()
        if data['success']:
            return data['predictions']
        else:
            return None
    except:
        return None

class CompositeOptimizer:
    """Multi-objective composite optimizer"""
    
    def __init__(self, target_property='tensile_strength', target_value=None, maximize=True):
        """
        Args:
            target_property: Property to optimize
            target_value: Target value (for constraint)
            maximize: True to maximize, False to minimize
        """
        self.target_property = target_property
        self.target_value = target_value
        self.maximize = maximize
        self.evaluation_count = 0
        self.best_value = float('-inf') if maximize else float('inf')
        self.best_config = None
    
    def objective_function(self, x):
        """Objective function for optimization"""
        self.evaluation_count += 1
        
        config = decode_configuration(x)
        pred = get_prediction(config)
        
        if pred is None:
            return 1e10
        
        value = pred[self.target_property]
        
        # Update best
        if self.maximize:
            if value > self.best_value:
                self.best_value = value
                self.best_config = config
            objective = -value  # Minimize negative = maximize
        else:
            if value < self.best_value:
                self.best_value = value
                self.best_config = config
            objective = value
        
        if self.evaluation_count % 10 == 0:
            print(f"  Evaluation {self.evaluation_count}: Best = {self.best_value:.2f}")
        
        return objective
    
    def optimize(self, max_evaluations=100):
        """Run optimization"""
        
        print("\n" + "="*60)
        print("🎯 COMPOSITE OPTIMIZATION")
        print("="*60)
        print(f"\nObjective: {'Maximize' if self.maximize else 'Minimize'} {self.target_property}")
        if self.target_value:
            print(f"Target: {self.target_value}")
        print(f"Max evaluations: {max_evaluations}\n")
        
        # Define bounds
        bounds = [
            (0, len(FIBER_NAMES) - 1),
            (0, len(MATRIX_NAMES) - 1),
            (0.30, 0.70),
            (0, len(LAYUP_NAMES) - 1),
            (0, len(MFG_NAMES) - 1)
        ]
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=max_evaluations // 15,
            popsize=15,
            seed=42,
            disp=False
        )
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*60)
        
        print(f"\nTotal evaluations: {self.evaluation_count}")
        print(f"Best {self.target_property}: {self.best_value:.2f}")
        
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"  Fiber: {self.best_config['fiber']}")
        print(f"  Matrix: {self.best_config['matrix']}")
        print(f"  Volume Fraction: {self.best_config['vf']:.2f}")
        print(f"  Layup: {self.best_config['layup']}")
        print(f"  Manufacturing: {self.best_config['manufacturing']}")
        
        # Get all properties for optimal config
        pred = get_prediction(self.best_config)
        if pred:
            print(f"\n📊 ALL PROPERTIES:")
            for prop, value in pred.items():
                print(f"  {prop}: {value:.2f}")
        
        return self.best_config, self.best_value

def example_maximize_strength():
    """Example: Maximize tensile strength"""
    
    optimizer = CompositeOptimizer(
        target_property='tensile_strength',
        maximize=True
    )
    
    config, value = optimizer.optimize(max_evaluations=150)
    return config, value

def example_minimize_cost():
    """Example: Minimize cost (using impact energy as proxy)"""
    
    optimizer = CompositeOptimizer(
        target_property='impact_energy',
        maximize=False
    )
    
    config, value = optimizer.optimize(max_evaluations=150)
    return config, value

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 COMPOSITE OPTIMIZATION")
    print("="*60)
    
    print("\n[1] Maximize Tensile Strength")
    print("[2] Maximize Impact Energy")
    print("[3] Maximize ILSS")
    
    choice = input("\nSelect optimization (1-3): ").strip()
    
    if choice == '1':
        example_maximize_strength()
    elif choice == '2':
        optimizer = CompositeOptimizer(target_property='impact_energy', maximize=True)
        optimizer.optimize(150)
    elif choice == '3':
        optimizer = CompositeOptimizer(target_property='ilss', maximize=True)
        optimizer.optimize(150)
    else:
        print("Invalid choice. Running example 1...")
        example_maximize_strength()