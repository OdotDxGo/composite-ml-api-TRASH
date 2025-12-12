"""
Cost Estimation System
Calculate manufacturing costs for composite materials
"""

import requests

API_URL = "http://localhost:5000"

# Material costs ($/kg)
MATERIAL_COSTS = {
    'E-Glass': {'fiber_cost': 2.5, 'density': 2.54},
    'Carbon T300': {'fiber_cost': 25.0, 'density': 1.76},
    'Kevlar 49': {'fiber_cost': 35.0, 'density': 1.44},
    'Basalt': {'fiber_cost': 3.5, 'density': 2.75},
    'Flax': {'fiber_cost': 1.8, 'density': 1.50},
    'Epoxy': {'matrix_cost': 8.0, 'density': 1.20},
    'Polyester': {'matrix_cost': 3.5, 'density': 1.15},
    'Vinyl Ester': {'matrix_cost': 6.0, 'density': 1.14},
    'PEEK': {'matrix_cost': 70.0, 'density': 1.32},
    'Polyamide 6': {'matrix_cost': 5.5, 'density': 1.14}
}

# Manufacturing costs ($/hr)
MANUFACTURING_COSTS = {
    'Hand Layup': {'labor_rate': 35.0, 'cycle_time': 4.0},
    'VARTM': {'labor_rate': 45.0, 'cycle_time': 3.0},
    'RTM': {'labor_rate': 50.0, 'cycle_time': 2.0},
    'Compression Molding': {'labor_rate': 40.0, 'cycle_time': 1.5},
    'Autoclave': {'labor_rate': 60.0, 'cycle_time': 6.0},
    'Filament Winding': {'labor_rate': 45.0, 'cycle_time': 2.5},
    'Pultrusion': {'labor_rate': 35.0, 'cycle_time': 0.5}
}

def calculate_cost(config, part_area=1.0, part_thickness=0.003, production_volume=100):
    """
    Calculate total part cost
    
    Args:
        config: Material configuration
        part_area: Part surface area (m²)
        part_thickness: Part thickness (m)
        production_volume: Number of parts
    """
    
    print("\n" + "="*60)
    print("💰 COST ESTIMATION")
    print("="*60)
    
    fiber = config['fiber']
    matrix = config['matrix']
    vf = config['vf']
    manufacturing = config['manufacturing']
    
    # Material properties
    rho_f = MATERIAL_COSTS[fiber]['density'] * 1000  # kg/m³
    rho_m = MATERIAL_COSTS[matrix]['density'] * 1000
    
    fiber_cost_per_kg = MATERIAL_COSTS[fiber]['fiber_cost']
    matrix_cost_per_kg = MATERIAL_COSTS[matrix]['matrix_cost']
    
    # Calculate masses
    part_volume = part_area * part_thickness  # m³
    rho_composite = vf * rho_f + (1 - vf) * rho_m
    part_mass = part_volume * rho_composite  # kg
    
    fiber_mass = part_mass * vf * (rho_f / rho_composite)
    matrix_mass = part_mass * (1 - vf) * (rho_m / rho_composite)
    
    # Material costs
    fiber_cost = fiber_mass * fiber_cost_per_kg
    matrix_cost = matrix_mass * matrix_cost_per_kg
    total_material_cost = fiber_cost + matrix_cost
    
    # Manufacturing costs
    labor_rate = MANUFACTURING_COSTS[manufacturing]['labor_rate']
    cycle_time = MANUFACTURING_COSTS[manufacturing]['cycle_time']
    
    labor_cost_per_part = (labor_rate * cycle_time * part_area) / production_volume
    
    # Total cost
    total_cost_per_part = total_material_cost + labor_cost_per_part
    
    # Get mechanical properties
    response = requests.post(f"{API_URL}/predict", json=config)
    if response.json()['success']:
        predictions = response.json()['predictions']
        
        print(f"\n📋 Configuration:")
        print(f"  Fiber: {fiber}")
        print(f"  Matrix: {matrix}")
        print(f"  Vf: {vf:.2f}")
        print(f"  Manufacturing: {manufacturing}")
        
        print(f"\n💵 Cost Breakdown (per part):")
        print(f"  Fiber material: ${fiber_cost:.2f}")
        print(f"  Matrix material: ${matrix_cost:.2f}")
        print(f"  Labor: ${labor_cost_per_part:.2f}")
        print(f"  ─────────────────")
        print(f"  TOTAL: ${total_cost_per_part:.2f}")
        print(f"  Batch ({production_volume} parts): ${total_cost_per_part * production_volume:.2f}")
        
        print(f"\n🎯 Mechanical Properties:")
        print(f"  Tensile Strength: {predictions['tensile_strength']:.1f} MPa")
        print(f"  Tensile Modulus: {predictions['tensile_modulus']:.1f} GPa")
        
        print(f"\n📈 Cost-Performance Metrics:")
        print(f"  Cost per MPa: ${total_cost_per_part / predictions['tensile_strength']:.4f}")
        print(f"  Cost per GPa: ${total_cost_per_part / predictions['tensile_modulus']:.2f}")
        
        return total_cost_per_part, predictions

if __name__ == "__main__":
    
    # Example configuration
    config = {
        'fiber': 'E-Glass',
        'matrix': 'Epoxy',
        'vf': 0.60,
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'VARTM'
    }
    
    calculate_cost(config, part_area=1.0, part_thickness=0.003, production_volume=100)