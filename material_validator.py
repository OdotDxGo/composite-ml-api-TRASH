"""
Advanced Material Validation System
For scientific composite design with intelligent recommendations
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    compatibility_score: float  # 0-100
    warnings: List[str]
    recommendations: List[str]
    alternative_configs: List[Dict]

class MaterialValidator:
    """
    Validates material combinations based on:
    - Physical compatibility
    - Manufacturing feasibility
    - Literature-backed constraints
    """
    
    # Material compatibility matrix (0-100 score)
    FIBER_MATRIX_COMPATIBILITY = {
        'E-Glass': {
            'Epoxy': 95,
            'Polyester': 90,
            'Vinyl Ester': 92,
            'PEEK': 75,
            'Polyamide 6': 85
        },
        'Carbon T300': {
            'Epoxy': 98,
            'Polyester': 40,  # Poor adhesion
            'Vinyl Ester': 85,
            'PEEK': 95,
            'Polyamide 6': 30  # Incompatible
        },
        'Kevlar 49': {
            'Epoxy': 95,
            'Polyester': 35,  # Poor interfacial bonding
            'Vinyl Ester': 88,
            'PEEK': 90,
            'Polyamide 6': 60
        },
        'Basalt': {
            'Epoxy': 88,
            'Polyester': 85,
            'Vinyl Ester': 92,
            'PEEK': 50,  # Rarely used
            'Polyamide 6': 80
        },
        'Flax': {
            'Epoxy': 85,
            'Polyester': 45,  # Moisture sensitivity
            'Vinyl Ester': 82,
            'PEEK': 20,  # Thermal degradation
            'Polyamide 6': 78
        }
    }
    
    # Manufacturing process compatibility
    MANUFACTURING_MATRIX_COMPATIBILITY = {
        'Autoclave': {
            'Epoxy': 98,
            'Polyester': 50,  # Overkill for polyester
            'Vinyl Ester': 85,
            'PEEK': 95,  # Required for PEEK
            'Polyamide 6': 70
        },
        'VARTM': {
            'Epoxy': 92,
            'Polyester': 88,
            'Vinyl Ester': 95,
            'PEEK': 20,  # Cannot process PEEK
            'Polyamide 6': 75
        },
        'RTM': {
            'Epoxy': 90,
            'Polyester': 85,
            'Vinyl Ester': 92,
            'PEEK': 25,
            'Polyamide 6': 80
        },
        'Compression Molding': {
            'Epoxy': 85,
            'Polyester': 90,
            'Vinyl Ester': 88,
            'PEEK': 92,  # Good for thermoplastics
            'Polyamide 6': 95
        },
        'Hand Layup': {
            'Epoxy': 88,
            'Polyester': 95,  # Traditional method
            'Vinyl Ester': 90,
            'PEEK': 0,  # Cannot process
            'Polyamide 6': 30
        },
        'Filament Winding': {
            'Epoxy': 95,
            'Polyester': 88,
            'Vinyl Ester': 92,
            'PEEK': 30,
            'Polyamide 6': 60
        },
        'Pultrusion': {
            'Epoxy': 90,
            'Polyester': 92,
            'Vinyl Ester': 95,
            'PEEK': 40,
            'Polyamide 6': 70
        }
    }
    
    # Vf limits by layup type (from literature)
    VF_LIMITS = {
        'Unidirectional 0°': {'min': 0.50, 'max': 0.75, 'optimal': 0.65},
        'Unidirectional 90°': {'min': 0.50, 'max': 0.75, 'optimal': 0.60},
        'Woven 0/90': {'min': 0.40, 'max': 0.65, 'optimal': 0.55},
        'Quasi-isotropic [0/45/90/-45]': {'min': 0.35, 'max': 0.60, 'optimal': 0.50},
        'Angle-ply [±45]': {'min': 0.35, 'max': 0.60, 'optimal': 0.48},
        'Cross-ply [0/90]': {'min': 0.40, 'max': 0.65, 'optimal': 0.55},
        'Random Mat': {'min': 0.25, 'max': 0.45, 'optimal': 0.35}
    }
    
    # Temperature compatibility (°C)
    TEMPERATURE_LIMITS = {
        'Epoxy': {'Tg': 120, 'max_service': 80, 'processing': 180},
        'Polyester': {'Tg': 90, 'max_service': 60, 'processing': 120},
        'Vinyl Ester': {'Tg': 110, 'max_service': 70, 'processing': 150},
        'PEEK': {'Tg': 143, 'max_service': 250, 'processing': 380},
        'Polyamide 6': {'Tg': 50, 'max_service': 80, 'processing': 230}
    }
    
    @staticmethod
    def validate_configuration(fiber: str, matrix: str, vf: float, 
                              layup: str, manufacturing: str) -> ValidationResult:
        """
        Comprehensive validation of composite configuration
        
        Returns ValidationResult with:
        - is_valid: bool
        - compatibility_score: 0-100
        - warnings: list of issues
        - recommendations: list of suggestions
        - alternative_configs: better options
        """
        
        warnings = []
        recommendations = []
        alternatives = []
        
        # 1. Check fiber-matrix compatibility
        fiber_matrix_score = MaterialValidator.FIBER_MATRIX_COMPATIBILITY[fiber][matrix]
        
        if fiber_matrix_score < 40:
            warnings.append(f"❌ CRITICAL: {fiber} and {matrix} have poor compatibility (score: {fiber_matrix_score}/100)")
            warnings.append(f"   Issue: Poor interfacial adhesion, expect delamination")
            is_valid = False
        elif fiber_matrix_score < 70:
            warnings.append(f"⚠️  {fiber} and {matrix} have limited compatibility (score: {fiber_matrix_score}/100)")
            warnings.append(f"   Recommendation: Consider alternative matrix")
            is_valid = True
        else:
            is_valid = True
        
        # 2. Check manufacturing compatibility
        mfg_matrix_score = MaterialValidator.MANUFACTURING_MATRIX_COMPATIBILITY[manufacturing][matrix]
        
        if mfg_matrix_score < 30:
            warnings.append(f"❌ CRITICAL: {manufacturing} cannot process {matrix}")
            if matrix == 'PEEK':
                warnings.append(f"   PEEK requires: Autoclave or Compression Molding (T>380°C)")
            is_valid = False
        elif mfg_matrix_score < 60:
            warnings.append(f"⚠️  {manufacturing} is suboptimal for {matrix} (score: {mfg_matrix_score}/100)")
        
        # 3. Check Vf limits
        vf_limits = MaterialValidator.VF_LIMITS[layup]
        
        if vf < vf_limits['min']:
            warnings.append(f"⚠️  Vf={vf:.2f} is below minimum ({vf_limits['min']:.2f}) for {layup}")
            warnings.append(f"   Risk: Insufficient reinforcement, matrix-dominated failure")
            recommendations.append(f"💡 Increase Vf to ≥{vf_limits['min']:.2f}")
        elif vf > vf_limits['max']:
            warnings.append(f"⚠️  Vf={vf:.2f} exceeds maximum ({vf_limits['max']:.2f}) for {layup}")
            warnings.append(f"   Risk: Fiber contact, poor wetting, voids")
            recommendations.append(f"💡 Decrease Vf to ≤{vf_limits['max']:.2f}")
        elif abs(vf - vf_limits['optimal']) > 0.10:
            recommendations.append(f"💡 Optimal Vf for {layup}: {vf_limits['optimal']:.2f} (current: {vf:.2f})")
        
        # 4. Fiber-specific warnings
        if fiber == 'Kevlar 49' and manufacturing == 'Autoclave':
            warnings.append(f"⚠️  Kevlar is sensitive to compression during autoclave processing")
            recommendations.append(f"💡 Consider VARTM or RTM instead")
        
        if fiber == 'Flax' and matrix in ['PEEK', 'Polyamide 6']:
            warnings.append(f"⚠️  Natural fiber ({fiber}) degrades at high processing temperatures")
            warnings.append(f"   {matrix} requires T>{MaterialValidator.TEMPERATURE_LIMITS[matrix]['processing']}°C")
            warnings.append(f"   Flax degrades at T>170°C")
        
        if fiber == 'Carbon T300' and matrix == 'Polyester':
            warnings.append(f"⚠️  Carbon-Polyester shows poor interfacial bonding")
            warnings.append(f"   Literature reports: 40-60% reduction in ILSS vs Carbon-Epoxy")
        
        # 5. Generate recommendations
        if fiber_matrix_score < 80:
            # Find better matrix
            better_matrices = []
            for mat, score in MaterialValidator.FIBER_MATRIX_COMPATIBILITY[fiber].items():
                if score > fiber_matrix_score + 10:
                    better_matrices.append((mat, score))
            
            if better_matrices:
                best_matrix = max(better_matrices, key=lambda x: x[1])
                recommendations.append(f"✅ Better matrix: {best_matrix[0]} (compatibility: {best_matrix[1]}/100)")
                alternatives.append({
                    'fiber': fiber,
                    'matrix': best_matrix[0],
                    'vf': vf_limits['optimal'],
                    'layup': layup,
                    'manufacturing': manufacturing,
                    'reason': f"Improved compatibility (+{best_matrix[1] - fiber_matrix_score} points)"
                })
        
        # 6. Calculate overall compatibility score
        overall_score = (fiber_matrix_score * 0.5 + mfg_matrix_score * 0.3 + 
                        (100 if vf_limits['min'] <= vf <= vf_limits['max'] else 50) * 0.2)
        
        # 7. Application-specific recommendations
        if fiber == 'E-Glass' and matrix in ['Epoxy', 'Vinyl Ester']:
            recommendations.append(f"✅ Excellent for: Marine, automotive, general purpose")
        
        if fiber == 'Carbon T300' and matrix == 'Epoxy' and manufacturing == 'Autoclave':
            recommendations.append(f"✅ Aerospace-grade configuration (highest performance)")
        
        if fiber == 'Flax' and matrix == 'Epoxy':
            recommendations.append(f"✅ Eco-friendly composite, suitable for: non-structural, interior panels")
        
        # 8. Cost warnings
        if fiber == 'Carbon T300' and manufacturing == 'Autoclave':
            warnings.append(f"💰 High-cost configuration (~$450/kg material + $200/hr processing)")
            recommendations.append(f"💡 Cost reduction: E-Glass/Epoxy + VARTM (~$85/kg)")
        
        return ValidationResult(
            is_valid=is_valid,
            compatibility_score=overall_score,
            warnings=warnings,
            recommendations=recommendations,
            alternative_configs=alternatives
        )
    
    @staticmethod
    def get_optimal_vf(layup: str) -> float:
        """Get optimal Vf for layup type"""
        return MaterialValidator.VF_LIMITS[layup]['optimal']
    
    @staticmethod
    def get_vf_range(layup: str) -> Tuple[float, float]:
        """Get valid Vf range for layup type"""
        limits = MaterialValidator.VF_LIMITS[layup]
        return (limits['min'], limits['max'])
    
    @staticmethod
    def get_best_combinations(top_n: int = 5) -> List[Dict]:
        """
        Get top N best material combinations based on compatibility scores
        """
        
        combinations = []
        
        for fiber in MaterialValidator.FIBER_MATRIX_COMPATIBILITY.keys():
            for matrix in MaterialValidator.FIBER_MATRIX_COMPATIBILITY[fiber].keys():
                for mfg in MaterialValidator.MANUFACTURING_MATRIX_COMPATIBILITY.keys():
                    
                    fiber_matrix_score = MaterialValidator.FIBER_MATRIX_COMPATIBILITY[fiber][matrix]
                    mfg_score = MaterialValidator.MANUFACTURING_MATRIX_COMPATIBILITY[mfg][matrix]
                    
                    if fiber_matrix_score >= 80 and mfg_score >= 70:
                        overall = (fiber_matrix_score + mfg_score) / 2
                        
                        combinations.append({
                            'fiber': fiber,
                            'matrix': matrix,
                            'manufacturing': mfg,
                            'compatibility_score': overall,
                            'fiber_matrix_score': fiber_matrix_score,
                            'manufacturing_score': mfg_score
                        })
        
        # Sort by overall score
        combinations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return combinations[:top_n]


if __name__ == "__main__":
    # Example usage
    validator = MaterialValidator()
    
    # Test case 1: Good configuration
    print("="*60)
    print("TEST 1: Carbon T300 / Epoxy / Autoclave")
    print("="*60)
    result = validator.validate_configuration(
        fiber='Carbon T300',
        matrix='Epoxy',
        vf=0.60,
        layup='Quasi-isotropic [0/45/90/-45]',
        manufacturing='Autoclave'
    )
    
    print(f"Valid: {result.is_valid}")
    print(f"Compatibility Score: {result.compatibility_score:.1f}/100")
    print(f"\nWarnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  {w}")
    print(f"\nRecommendations: {len(result.recommendations)}")
    for r in result.recommendations:
        print(f"  {r}")
    
    # Test case 2: Bad configuration
    print("\n" + "="*60)
    print("TEST 2: Carbon T300 / Polyester / Hand Layup (BAD)")
    print("="*60)
    result = validator.validate_configuration(
        fiber='Carbon T300',
        matrix='Polyester',
        vf=0.80,  # Too high
        layup='Quasi-isotropic [0/45/90/-45]',
        manufacturing='Hand Layup'
    )
    
    print(f"Valid: {result.is_valid}")
    print(f"Compatibility Score: {result.compatibility_score:.1f}/100")
    print(f"\nWarnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  {w}")
    print(f"\nRecommendations: {len(result.recommendations)}")
    for r in result.recommendations:
        print(f"  {r}")
    
    # Best combinations
    print("\n" + "="*60)
    print("TOP 5 BEST CONFIGURATIONS")
    print("="*60)
    best = validator.get_best_combinations(top_n=5)
    for i, config in enumerate(best, 1):
        print(f"\n{i}. {config['fiber']} / {config['matrix']} / {config['manufacturing']}")
        print(f"   Overall Score: {config['compatibility_score']:.1f}/100")
