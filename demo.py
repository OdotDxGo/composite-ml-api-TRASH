"""
Demo Script - Test All Features
Tests material validation, plotting, PDF generation, and simulation
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from material_validator import MaterialValidator
from scientific_plotter import ScientificPlotter
from pdf_report_generator import ScientificReportGenerator
from mechanical_simulator import MechanicalSimulator, StressState

def demo_validation():
    """Demo: Material Validation"""
    print("\n" + "="*70)
    print("🔍 DEMO 1: MATERIAL VALIDATION")
    print("="*70)
    
    validator = MaterialValidator()
    
    # Test 1: Good configuration
    print("\n📋 Test 1: Carbon T300 / Epoxy / Autoclave (GOOD)")
    result = validator.validate_configuration(
        fiber='Carbon T300',
        matrix='Epoxy',
        vf=0.60,
        layup='Quasi-isotropic [0/45/90/-45]',
        manufacturing='Autoclave'
    )
    
    print(f"✓ Valid: {result.is_valid}")
    print(f"✓ Compatibility Score: {result.compatibility_score:.1f}/100")
    print(f"✓ Warnings: {len(result.warnings)}")
    for w in result.warnings[:3]:
        print(f"  {w}")
    print(f"✓ Recommendations: {len(result.recommendations)}")
    for r in result.recommendations[:3]:
        print(f"  {r}")
    
    # Test 2: Bad configuration
    print("\n📋 Test 2: Carbon T300 / Polyester / Hand Layup (BAD)")
    result = validator.validate_configuration(
        fiber='Carbon T300',
        matrix='Polyester',
        vf=0.80,  # Too high!
        layup='Quasi-isotropic [0/45/90/-45]',
        manufacturing='Hand Layup'
    )
    
    print(f"✗ Valid: {result.is_valid}")
    print(f"✗ Compatibility Score: {result.compatibility_score:.1f}/100")
    print(f"✗ Warnings: {len(result.warnings)}")
    for w in result.warnings[:5]:
        print(f"  {w}")

def demo_plotting():
    """Demo: Scientific Plotting"""
    print("\n" + "="*70)
    print("📊 DEMO 2: SCIENTIFIC PLOTTING")
    print("="*70)
    
    plotter = ScientificPlotter()
    os.makedirs('output/plots', exist_ok=True)
    
    # Demo data
    print("\n📈 Generating plots...")
    
    # 1. Ashby Chart
    sample_data = [
        {'fiber': 'E-Glass', 'matrix': 'Polyester', 'density': 2.0, 'tensile_strength': 230},
        {'fiber': 'Carbon T300', 'matrix': 'Epoxy', 'density': 1.55, 'tensile_strength': 1420},
        {'fiber': 'Kevlar 49', 'matrix': 'Epoxy', 'density': 1.38, 'tensile_strength': 1280},
        {'fiber': 'Basalt', 'matrix': 'Vinyl Ester', 'density': 2.1, 'tensile_strength': 480},
    ]
    
    plotter.plot_ashby_chart(sample_data, 'output/plots/ashby_chart.png')
    print("  ✓ Ashby Chart saved: output/plots/ashby_chart.png")
    
    # 2. Vf Sensitivity
    import numpy as np
    vf_values = np.linspace(0.30, 0.70, 21)
    properties = {
        'tensile_strength': 300 + 800 * vf_values + np.random.normal(0, 20, len(vf_values)),
        'tensile_modulus': 10 + 60 * vf_values + np.random.normal(0, 2, len(vf_values)),
        'ilss': 20 + 30 * vf_values + np.random.normal(0, 1, len(vf_values)),
        'impact_energy': 10 + 20 * vf_values + np.random.normal(0, 0.5, len(vf_values)),
    }
    
    plotter.plot_vf_sensitivity(vf_values, properties, 'output/plots/vf_sensitivity.png')
    print("  ✓ Vf Sensitivity saved: output/plots/vf_sensitivity.png")
    
    # 3. Stress Distribution
    plotter.plot_stress_distribution(save_path='output/plots/stress_distribution.png')
    print("  ✓ Stress Distribution saved: output/plots/stress_distribution.png")
    
    # 4. Failure Envelope
    config = {'tensile_strength': 678, 'compressive_strength': 512}
    plotter.plot_failure_envelope(config, 'output/plots/failure_envelope.png')
    print("  ✓ Failure Envelope saved: output/plots/failure_envelope.png")
    
    print("\n✅ All plots generated at 300 DPI!")

def demo_pdf_report():
    """Demo: PDF Report Generation"""
    print("\n" + "="*70)
    print("📄 DEMO 3: PDF REPORT GENERATION")
    print("="*70)
    
    config = {
        'fiber': 'Carbon T300',
        'matrix': 'Epoxy',
        'vf': 0.60,
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'Autoclave'
    }
    
    predictions = {
        'tensile_strength': {'value': 678.4, 'lower': 645.2, 'upper': 711.6, 'std': 17.2},
        'tensile_modulus': {'value': 68.2, 'lower': 65.1, 'upper': 71.3, 'std': 1.6},
        'compressive_strength': {'value': 512.7, 'lower': 485.3, 'upper': 540.1, 'std': 14.2},
        'flexural_strength': {'value': 834.5, 'lower': 795.8, 'upper': 873.2, 'std': 20.0},
        'flexural_modulus': {'value': 71.3, 'lower': 68.0, 'upper': 74.6, 'std': 1.7},
        'ilss': {'value': 52.4, 'lower': 49.8, 'upper': 55.0, 'std': 1.3},
        'impact_energy': {'value': 28.7, 'lower': 27.2, 'upper': 30.2, 'std': 0.8}
    }
    
    validator = MaterialValidator()
    validation_result = validator.validate_configuration(
        config['fiber'], config['matrix'], config['vf'], 
        config['layup'], config['manufacturing']
    )
    
    print("\n📝 Generating scientific PDF report...")
    
    generator = ScientificReportGenerator()
    output_path = 'output/reports/composite_analysis_report.pdf'
    os.makedirs('output/reports', exist_ok=True)
    
    generator.generate_report(
        config=config,
        predictions=predictions,
        validation_result=validation_result,
        plots_dir='output/plots',
        output_path=output_path
    )
    
    print(f"✅ PDF Report generated: {output_path}")
    print("\n📋 Report Contents:")
    print("  1. Title Page")
    print("  2. Executive Summary")
    print("  3. Material Configuration")
    print("  4. Predicted Mechanical Properties")
    print("  5. Statistical Analysis")
    print("  6. Recommendations")
    print("  7. Graphical Analysis")
    print("  8. References")

def demo_mechanical_simulation():
    """Demo: Mechanical Simulation"""
    print("\n" + "="*70)
    print("🔬 DEMO 4: MECHANICAL SIMULATION")
    print("="*70)
    
    config = {
        'fiber': 'Carbon T300',
        'matrix': 'Epoxy',
        'vf': 0.60,
        'layup': 'Quasi-isotropic [0/45/90/-45]',
        'manufacturing': 'Autoclave'
    }
    
    predictions = {
        'tensile_strength': {'value': 678.4},
        'compressive_strength': {'value': 512.7},
        'ilss': {'value': 52.4}
    }
    
    # 1. Stress Distribution
    print("\n📊 1. Stress Distribution Analysis")
    stress_dist = MechanicalSimulator.calculate_stress_distribution(
        config, applied_load=100, load_type='tension'
    )
    print(f"  Applied Load: 100 MPa")
    print(f"  Fiber Stress: {stress_dist['stress_fiber']:.1f} MPa")
    print(f"  Matrix Stress: {stress_dist['stress_matrix']:.1f} MPa")
    print(f"  Interface Stress: {stress_dist['stress_interface']:.1f} MPa (stress concentration)")
    
    # 2. Failure Analysis
    print("\n⚡ 2. Tsai-Wu Failure Analysis")
    stress_state = StressState(sigma_x=300, sigma_y=50, tau_xy=20)
    failure = MechanicalSimulator.tsai_wu_failure_analysis(
        stress_state, config, predictions
    )
    print(f"  Applied Stress: σ_x={stress_state.sigma_x} MPa, σ_y={stress_state.sigma_y} MPa, τ_xy={stress_state.tau_xy} MPa")
    print(f"  Failure Index: {failure.failure_index:.3f} {'(SAFE)' if failure.failure_index < 1 else '(FAILURE)'}")
    print(f"  Safety Factor: {failure.safety_factor:.2f}")
    print(f"  Failure Mode: {failure.failure_mode}")
    
    # 3. Progressive Damage
    print("\n🔄 3. Progressive Damage Simulation")
    damage_history = MechanicalSimulator.progressive_damage_simulation(
        config, predictions, max_stress=700, num_steps=5
    )
    
    print("\n  Step | Stress (MPa) | Damage | Safety Factor | Status")
    print("  " + "-"*60)
    for state in damage_history:
        status = "❌ FAILED" if state['failed'] else "✓ Safe"
        print(f"  {state['step']:4d} | {state['stress']:12.1f} | {state['damage']:6.3f} | "
              f"{state['safety_factor']:13.2f} | {status}")
    
    print("\n✅ All simulations complete!")

def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("🎓 HYBRID PIRF SYSTEM - COMPLETE DEMO")
    print("Scientific Edition for Doctoral Research")
    print("="*70)
    
    try:
        demo_validation()
        demo_plotting()
        demo_pdf_report()
        demo_mechanical_simulation()
        
        print("\n" + "="*70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📁 Generated Files:")
        print("  output/plots/ashby_chart.png")
        print("  output/plots/vf_sensitivity.png")
        print("  output/plots/stress_distribution.png")
        print("  output/plots/failure_envelope.png")
        print("  output/reports/composite_analysis_report.pdf")
        
        print("\n💡 Next Steps:")
        print("  1. Review generated plots in output/plots/")
        print("  2. Open PDF report in output/reports/")
        print("  3. Integrate into app.py and deploy to Railway")
        print("  4. Use for your dissertation and Scopus papers!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
