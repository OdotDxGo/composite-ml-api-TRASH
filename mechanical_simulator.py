"""
Mechanical Simulation Module
Advanced stress analysis and failure prediction for composites
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class StressState:
    """Stress state at a point"""
    sigma_x: float  # Longitudinal stress (MPa)
    sigma_y: float  # Transverse stress (MPa)
    tau_xy: float   # Shear stress (MPa)

@dataclass
class FailureAnalysis:
    """Failure analysis results"""
    failure_index: float
    safety_factor: float
    failure_mode: str
    critical_stress: float
    will_fail: bool

class MechanicalSimulator:
    """
    Performs mechanical simulations for composite materials:
    - Stress distribution analysis
    - Failure prediction (Tsai-Wu, Maximum Stress)
    - Safety factor calculation
    - Progressive damage modeling
    """
    
    @staticmethod
    def calculate_stress_distribution(
        config: Dict, 
        applied_load: float,
        load_type: str = 'tension'
    ) -> Dict[str, np.ndarray]:
        """
        Calculate stress distribution in composite
        
        Args:
            config: Material configuration
            applied_load: Applied stress (MPa)
            load_type: 'tension', 'compression', or 'shear'
        
        Returns:
            Dictionary with stress fields
        """
        
        # Get properties
        vf = config.get('vf', 0.55)
        E_f = MechanicalSimulator._get_fiber_modulus(config['fiber'])
        E_m = MechanicalSimulator._get_matrix_modulus(config['matrix'])
        
        # Rule of mixtures for stiffness
        E_L = E_f * vf + E_m * (1 - vf)
        
        # Stress partitioning
        # Fiber carries more load due to higher stiffness
        stress_fiber = applied_load * (E_f / E_L)
        stress_matrix = applied_load * (E_m / E_L)
        
        # Stress concentration at fiber-matrix interface
        # Typically 1.5-2.0x higher than average
        stress_interface = applied_load * 1.8
        
        # Create mesh
        nx, ny = 100, 100
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Simulate fiber locations
        np.random.seed(42)
        num_fibers = int(vf * 50)
        fiber_x = np.random.uniform(0.1, 0.9, num_fibers)
        fiber_y = np.random.uniform(0.1, 0.9, num_fibers)
        fiber_radius = 0.02
        
        # Initialize stress field (matrix stress)
        stress_field = np.ones_like(X) * stress_matrix
        
        # Add fiber stress
        for fx, fy in zip(fiber_x, fiber_y):
            dist = np.sqrt((X - fx)**2 + (Y - fy)**2)
            
            # Fiber region
            fiber_mask = dist < fiber_radius
            stress_field[fiber_mask] = stress_fiber
            
            # Interface region (stress concentration)
            interface_mask = (dist >= fiber_radius) & (dist < fiber_radius * 1.2)
            stress_field[interface_mask] = stress_interface
        
        return {
            'X': X,
            'Y': Y,
            'stress_field': stress_field,
            'stress_fiber': stress_fiber,
            'stress_matrix': stress_matrix,
            'stress_interface': stress_interface,
            'fiber_locations': list(zip(fiber_x, fiber_y)),
            'fiber_radius': fiber_radius
        }
    
    @staticmethod
    def tsai_wu_failure_analysis(
        stress_state: StressState,
        config: Dict,
        predictions: Dict
    ) -> FailureAnalysis:
        """
        Tsai-Wu failure criterion analysis
        
        F_i * σ_i + F_ij * σ_i * σ_j < 1 (no failure)
        
        Returns FailureAnalysis with failure index and safety factor
        """
        
        # Get strength values
        sigma_xt = predictions.get('tensile_strength', {}).get('value', 300)
        sigma_xc = predictions.get('compressive_strength', {}).get('value', 200)
        sigma_yt = sigma_xt * 0.15  # Transverse strength (typically 10-20% of longitudinal)
        sigma_yc = sigma_xc * 0.20
        tau_xy_ult = predictions.get('ilss', {}).get('value', 30)
        
        # Tsai-Wu coefficients
        F1 = 1/sigma_xt - 1/sigma_xc
        F2 = 1/sigma_yt - 1/sigma_yc
        
        F11 = 1/(sigma_xt * sigma_xc)
        F22 = 1/(sigma_yt * sigma_yc)
        F66 = 1/(tau_xy_ult ** 2)
        
        # Interaction term (typically -0.5 * sqrt(F11 * F22))
        F12 = -0.5 * np.sqrt(F11 * F22)
        
        # Calculate failure index
        FI = (F1 * stress_state.sigma_x + 
              F2 * stress_state.sigma_y +
              F11 * stress_state.sigma_x ** 2 +
              F22 * stress_state.sigma_y ** 2 +
              F66 * stress_state.tau_xy ** 2 +
              2 * F12 * stress_state.sigma_x * stress_state.sigma_y)
        
        # Safety factor (inverse of failure index)
        if FI > 0:
            safety_factor = 1 / np.sqrt(FI)
        else:
            safety_factor = float('inf')
        
        # Determine failure mode
        if abs(stress_state.sigma_x) / sigma_xt > 0.8:
            failure_mode = "Fiber tensile failure"
        elif abs(stress_state.sigma_x) / sigma_xc > 0.8:
            failure_mode = "Fiber compressive failure"
        elif abs(stress_state.tau_xy) / tau_xy_ult > 0.8:
            failure_mode = "Matrix shear failure"
        elif abs(stress_state.sigma_y) / sigma_yt > 0.8:
            failure_mode = "Matrix transverse failure"
        else:
            failure_mode = "No dominant mode"
        
        will_fail = FI >= 1.0
        
        return FailureAnalysis(
            failure_index=FI,
            safety_factor=safety_factor,
            failure_mode=failure_mode,
            critical_stress=max(abs(stress_state.sigma_x), 
                              abs(stress_state.sigma_y), 
                              abs(stress_state.tau_xy)),
            will_fail=will_fail
        )
    
    @staticmethod
    def maximum_stress_criterion(
        stress_state: StressState,
        predictions: Dict
    ) -> FailureAnalysis:
        """
        Maximum Stress failure criterion (simpler, more conservative)
        
        Failure occurs when any stress component exceeds its allowable
        """
        
        sigma_xt = predictions.get('tensile_strength', {}).get('value', 300)
        sigma_xc = predictions.get('compressive_strength', {}).get('value', 200)
        sigma_yt = sigma_xt * 0.15
        sigma_yc = sigma_xc * 0.20
        tau_xy_ult = predictions.get('ilss', {}).get('value', 30)
        
        # Check each stress component
        ratios = [
            abs(stress_state.sigma_x) / sigma_xt if stress_state.sigma_x > 0 else abs(stress_state.sigma_x) / sigma_xc,
            abs(stress_state.sigma_y) / sigma_yt if stress_state.sigma_y > 0 else abs(stress_state.sigma_y) / sigma_yc,
            abs(stress_state.tau_xy) / tau_xy_ult
        ]
        
        failure_index = max(ratios)
        safety_factor = 1 / failure_index if failure_index > 0 else float('inf')
        
        # Determine failure mode
        max_ratio_idx = np.argmax(ratios)
        failure_modes = [
            "Longitudinal failure",
            "Transverse failure",
            "Shear failure"
        ]
        failure_mode = failure_modes[max_ratio_idx]
        
        will_fail = failure_index >= 1.0
        
        return FailureAnalysis(
            failure_index=failure_index,
            safety_factor=safety_factor,
            failure_mode=failure_mode,
            critical_stress=max(abs(stress_state.sigma_x), 
                              abs(stress_state.sigma_y), 
                              abs(stress_state.tau_xy)),
            will_fail=will_fail
        )
    
    @staticmethod
    def calculate_failure_envelope(
        predictions: Dict,
        num_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Generate Tsai-Wu failure envelope
        
        Returns mesh of stress states and failure indices
        """
        
        sigma_xt = predictions.get('tensile_strength', {}).get('value', 300)
        sigma_xc = predictions.get('compressive_strength', {}).get('value', 200)
        
        # Create stress grid
        sigma_x = np.linspace(-sigma_xc * 1.2, sigma_xt * 1.2, num_points)
        sigma_y = np.linspace(-sigma_xc * 1.2, sigma_xt * 1.2, num_points)
        X, Y = np.meshgrid(sigma_x, sigma_y)
        
        # Calculate failure index at each point
        FI = np.zeros_like(X)
        
        for i in range(num_points):
            for j in range(num_points):
                stress_state = StressState(
                    sigma_x=X[i, j],
                    sigma_y=Y[i, j],
                    tau_xy=0
                )
                
                # Simplified Tsai-Wu
                sigma_yt = sigma_xt * 0.15
                sigma_yc = sigma_xc * 0.20
                
                F1 = 1/sigma_xt - 1/sigma_xc
                F2 = 1/sigma_yt - 1/sigma_yc
                F11 = 1/(sigma_xt * sigma_xc)
                F22 = 1/(sigma_yt * sigma_yc)
                F12 = -0.5 * np.sqrt(F11 * F22)
                
                FI[i, j] = (F1 * stress_state.sigma_x + 
                           F2 * stress_state.sigma_y +
                           F11 * stress_state.sigma_x ** 2 +
                           F22 * stress_state.sigma_y ** 2 +
                           2 * F12 * stress_state.sigma_x * stress_state.sigma_y)
        
        return {
            'sigma_x': X,
            'sigma_y': Y,
            'failure_index': FI,
            'safe_region': FI < 1.0
        }
    
    @staticmethod
    def progressive_damage_simulation(
        config: Dict,
        predictions: Dict,
        max_stress: float,
        num_steps: int = 10
    ) -> List[Dict]:
        """
        Simulate progressive damage under increasing load
        
        Returns list of damage states
        """
        
        damage_history = []
        
        for step in range(num_steps + 1):
            current_stress = max_stress * (step / num_steps)
            
            stress_state = StressState(
                sigma_x=current_stress,
                sigma_y=current_stress * 0.1,  # Small transverse stress
                tau_xy=0
            )
            
            failure_analysis = MechanicalSimulator.tsai_wu_failure_analysis(
                stress_state, config, predictions
            )
            
            # Damage metric (0 = intact, 1 = failed)
            damage = min(failure_analysis.failure_index, 1.0)
            
            damage_state = {
                'step': step,
                'stress': current_stress,
                'damage': damage,
                'failure_index': failure_analysis.failure_index,
                'safety_factor': failure_analysis.safety_factor,
                'mode': failure_analysis.failure_mode,
                'failed': failure_analysis.will_fail
            }
            
            damage_history.append(damage_state)
        
        return damage_history
    
    @staticmethod
    def _get_fiber_modulus(fiber: str) -> float:
        """Get fiber elastic modulus (GPa)"""
        moduli = {
            'E-Glass': 73,
            'Carbon T300': 230,
            'Kevlar 49': 131,
            'Basalt': 89,
            'Flax': 58
        }
        return moduli.get(fiber, 73)
    
    @staticmethod
    def _get_matrix_modulus(matrix: str) -> float:
        """Get matrix elastic modulus (GPa)"""
        moduli = {
            'Epoxy': 3.2,
            'Polyester': 3.5,
            'Vinyl Ester': 3.4,
            'PEEK': 3.9,
            'Polyamide 6': 2.8
        }
        return moduli.get(matrix, 3.2)


# Example usage
if __name__ == "__main__":
    
    config = {
        'fiber': 'Carbon T300',
        'matrix': 'Epoxy',
        'vf': 0.60,
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'Autoclave'
    }
    
    predictions = {
        'tensile_strength': {'value': 678.4, 'std': 17.2},
        'compressive_strength': {'value': 512.7, 'std': 14.2},
        'ilss': {'value': 52.4, 'std': 1.3}
    }
    
    print("="*60)
    print("MECHANICAL SIMULATION EXAMPLE")
    print("="*60)
    
    # Stress distribution
    print("\n1. Stress Distribution Analysis")
    stress_dist = MechanicalSimulator.calculate_stress_distribution(
        config, applied_load=100, load_type='tension'
    )
    print(f"   Fiber stress: {stress_dist['stress_fiber']:.1f} MPa")
    print(f"   Matrix stress: {stress_dist['stress_matrix']:.1f} MPa")
    print(f"   Interface stress: {stress_dist['stress_interface']:.1f} MPa")
    
    # Failure analysis
    print("\n2. Tsai-Wu Failure Analysis")
    stress_state = StressState(sigma_x=300, sigma_y=50, tau_xy=20)
    failure = MechanicalSimulator.tsai_wu_failure_analysis(
        stress_state, config, predictions
    )
    print(f"   Failure Index: {failure.failure_index:.3f}")
    print(f"   Safety Factor: {failure.safety_factor:.2f}")
    print(f"   Failure Mode: {failure.failure_mode}")
    print(f"   Will Fail: {failure.will_fail}")
    
    # Progressive damage
    print("\n3. Progressive Damage Simulation")
    damage_history = MechanicalSimulator.progressive_damage_simulation(
        config, predictions, max_stress=700, num_steps=5
    )
    
    print("   Step | Stress (MPa) | Damage | SF    | Status")
    print("   " + "-"*50)
    for state in damage_history:
        status = "❌ FAILED" if state['failed'] else "✓ Safe"
        print(f"   {state['step']:4d} | {state['stress']:12.1f} | {state['damage']:6.3f} | "
              f"{state['safety_factor']:5.2f} | {status}")
    
    print("\n✓ Simulation complete!")
