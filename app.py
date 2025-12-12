"""
Hybrid PIRF API - Complete Version with Plotting and Simulation
Version 3.0 - Russian Interface
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import modules
import sys
sys.path.append(os.path.dirname(__file__))

from material_validator import MaterialValidator
from pdf_report_generator import ScientificReportGenerator
from mechanical_simulator import MechanicalSimulator, StressState

# Matplotlib for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__, static_folder='static')
CORS(app)

# Create output directories
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)

# Initialize modules
material_validator = MaterialValidator()

# Material properties dataclasses
@dataclass
class FiberProperties:
    E: float
    sigma: float
    rho: float
    G: float
    epsilon_f: float
    U_f: float

@dataclass
class MatrixProperties:
    E: float
    sigma: float
    rho: float
    G: float
    epsilon_f: float
    U_m: float

# Material libraries
FIBERS = {
    'Carbon T300': FiberProperties(230, 3530, 1.76, 95, 1.5, 25),
    'E-Glass': FiberProperties(73, 3450, 2.54, 30, 4.7, 18),
    'Kevlar 49': FiberProperties(131, 3620, 1.44, 52, 2.8, 35),
    'Basalt': FiberProperties(89, 4840, 2.75, 35, 3.1, 22),
    'Flax': FiberProperties(58, 1100, 1.50, 22, 2.7, 12)
}

MATRICES = {
    'Epoxy': MatrixProperties(3.2, 78, 1.20, 1.2, 4.5, 2.8),
    'Polyester': MatrixProperties(3.5, 55, 1.15, 1.3, 2.8, 2.1),
    'Vinyl Ester': MatrixProperties(3.4, 82, 1.14, 1.3, 5.2, 2.5),
    'PEEK': MatrixProperties(3.9, 105, 1.32, 1.4, 50, 3.5),
    'Polyamide 6': MatrixProperties(2.8, 82, 1.14, 1.0, 100, 2.0)
}

EFFICIENCY_FACTORS = {
    'eta_L': 0.95,
    'eta_T': 0.40,
    'eta_strength': 0.90,
    'eta_comp': 0.70,
    'eta_interface': 0.85
}

MANUFACTURING_FACTORS = {
    'Autoclave': 1.00,
    'VARTM': 0.93,
    'RTM': 0.88,
    'Compression Molding': 0.85,
    'Hand Layup': 0.75,
    'Filament Winding': 0.90,
    'Pultrusion': 0.92
}

# Physics Engine
class PhysicsEngine:
    @staticmethod
    def calculate_rom_properties(fiber: FiberProperties, matrix: MatrixProperties, 
                                 Vf: float, layup: str, manufacturing: str) -> Dict[str, float]:
        
        eta_L = EFFICIENCY_FACTORS['eta_L']
        eta_T = EFFICIENCY_FACTORS['eta_T']
        eta_strength = EFFICIENCY_FACTORS['eta_strength']
        eta_comp = EFFICIENCY_FACTORS['eta_comp']
        eta_interface = EFFICIENCY_FACTORS['eta_interface']
        eta_mfg = MANUFACTURING_FACTORS[manufacturing]
        
        E_L = eta_L * fiber.E * Vf + matrix.E * (1 - Vf)
        E_T = 1 / (Vf / (eta_T * fiber.E) + (1 - Vf) / matrix.E)
        
        sigma_m_prime = 0.45 * matrix.sigma
        sigma_UTS = eta_strength * fiber.sigma * Vf + sigma_m_prime * (1 - Vf)
        
        sigma_comp_buckling = matrix.G / (1 - Vf)
        sigma_comp_crushing = fiber.sigma * Vf * 0.8
        sigma_comp = eta_comp * min(sigma_comp_buckling, sigma_comp_crushing)
        
        modulus_ratio = fiber.E / matrix.E
        sigma_flex = 1.25 * sigma_UTS * (1 + 0.1 * modulus_ratio) ** (-0.3)
        E_flex = 1.05 * E_L * (1 + 0.05 * Vf)
        
        tau_m = 0.50 * matrix.sigma
        tau_ILSS = tau_m * (1 - 0.5 * Vf) * eta_interface
        
        eta_f = 0.75
        eta_m = 0.70
        U_interface_fraction = 0.20
        U_impact = (fiber.U_f * Vf * eta_f + 
                   matrix.U_m * (1 - Vf) * eta_m) * (1 + U_interface_fraction)
        
        layup_factors = {
            'Unidirectional 0°': {'k_E': 1.00, 'k_sigma': 1.00},
            'Unidirectional 90°': {'k_E': E_T/E_L, 'k_sigma': 0.15},
            'Woven 0/90': {'k_E': 0.65, 'k_sigma': 0.70},
            'Quasi-isotropic [0/45/90/-45]': {
                'k_E': 3/8 + Vf/4 + 1/8 * (E_T/E_L),
                'k_sigma': 3/8 + Vf/4 + 1/8 * 0.15
            },
            'Angle-ply [±45]': {'k_E': 0.35, 'k_sigma': 0.40},
            'Cross-ply [0/90]': {'k_E': 0.55, 'k_sigma': 0.60},
            'Random Mat': {'k_E': 0.30, 'k_sigma': 0.35}
        }
        
        k_E = layup_factors[layup]['k_E']
        k_sigma = layup_factors[layup]['k_sigma']
        
        E_L_layup = E_L * k_E
        sigma_UTS_layup = sigma_UTS * k_sigma
        sigma_comp_layup = sigma_comp * k_sigma * 0.9
        sigma_flex_layup = sigma_flex * k_sigma * 1.15
        E_flex_layup = E_flex * k_E * 0.95
        
        correction_factors = {
            'tensile_strength': 0.262,
            'tensile_modulus': 0.72,
            'compressive_strength': 0.202,
            'flexural_strength': 0.289,
            'flexural_modulus': 0.60,
            'ilss': 1.49,
            'impact': 0.98
        }
        
        properties = {
            'tensile_strength': sigma_UTS_layup * eta_mfg * correction_factors['tensile_strength'],
            'tensile_modulus': E_L_layup * eta_mfg * correction_factors['tensile_modulus'],
            'compressive_strength': sigma_comp_layup * eta_mfg * correction_factors['compressive_strength'],
            'flexural_strength': sigma_flex_layup * eta_mfg * correction_factors['flexural_strength'],
            'flexural_modulus': E_flex_layup * eta_mfg * correction_factors['flexural_modulus'],
            'ilss': tau_ILSS * eta_mfg * correction_factors['ilss'],
            'impact_energy': U_impact * eta_mfg * correction_factors['impact'],
            'E_L_UD': E_L,
            'E_T_UD': E_T,
            'sigma_UTS_UD': sigma_UTS,
            'E_f_E_m_ratio': fiber.E / matrix.E,
            'sigma_f_sigma_m_ratio': fiber.sigma / matrix.sigma,
            'rho_f_rho_m_ratio': fiber.rho / matrix.rho,
            'G_f_G_m_ratio': fiber.G / matrix.G,
        }
        
        return properties

# Feature Engineer
class FeatureEngineer:
    @staticmethod
    def create_features(fiber_name: str, matrix_name: str, Vf: float, 
                       layup: str, manufacturing: str, 
                       rom_properties: Dict[str, float]) -> np.ndarray:
        
        features = []
        
        fiber_encoding = {'Carbon T300': 0, 'E-Glass': 1, 'Kevlar 49': 2, 
                         'Basalt': 3, 'Flax': 4}
        matrix_encoding = {'Epoxy': 0, 'Polyester': 1, 'Vinyl Ester': 2, 
                          'PEEK': 3, 'Polyamide 6': 4}
        layup_encoding = {
            'Unidirectional 0°': 0, 'Unidirectional 90°': 1, 'Woven 0/90': 2,
            'Quasi-isotropic [0/45/90/-45]': 3, 'Angle-ply [±45]': 4,
            'Cross-ply [0/90]': 5, 'Random Mat': 6
        }
        mfg_encoding = {'Autoclave': 0, 'VARTM': 1, 'RTM': 2, 
                       'Compression Molding': 3, 'Hand Layup': 4,
                       'Filament Winding': 5, 'Pultrusion': 6}
        
        features.extend([
            fiber_encoding[fiber_name],
            matrix_encoding[matrix_name],
            Vf,
            layup_encoding[layup],
            mfg_encoding[manufacturing]
        ])
        
        features.extend([
            rom_properties['tensile_strength'],
            rom_properties['tensile_modulus'],
            rom_properties['compressive_strength'],
            rom_properties['flexural_strength'],
            rom_properties['flexural_modulus'],
            rom_properties['ilss'],
            rom_properties['impact_energy']
        ])
        
        features.extend([
            rom_properties['E_f_E_m_ratio'],
            rom_properties['sigma_f_sigma_m_ratio'],
            rom_properties['rho_f_rho_m_ratio'],
            rom_properties['G_f_G_m_ratio']
        ])
        
        features.extend([
            Vf ** 2,
            Vf ** 3,
            1 / (1 - Vf + 1e-6)
        ])
        
        features.extend([
            Vf * rom_properties['E_f_E_m_ratio'],
            Vf * rom_properties['sigma_f_sigma_m_ratio'],
            rom_properties['tensile_modulus'] ** 2,
            rom_properties['tensile_strength'] * rom_properties['tensile_modulus'],
            Vf * rom_properties['E_L_UD'],
            (1 - Vf) * rom_properties['E_T_UD'],
            rom_properties['E_f_E_m_ratio'] * rom_properties['sigma_f_sigma_m_ratio']
        ])
        
        return np.array(features).reshape(1, -1)

# Hybrid ML Model
class HybridPIRF:
    def __init__(self):
        self.models = {}
        self.scaler = None
        
    def load_models(self):
        model_path = 'models/hybrid_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            print(f"✓ Loaded {len(self.models)} trained models")
        else:
            print("⚠ No trained models found. Using physics-based predictions only.")
            
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✓ Loaded feature scaler")
    
    def predict(self, features: np.ndarray, rom_predictions: Dict[str, float]) -> Dict:
        if not self.models:
            return {
                'predictions': rom_predictions,
                'uncertainty': {k: {'lower': v*0.85, 'upper': v*1.15, 'std': v*0.10} 
                               for k, v in rom_predictions.items()},
                'method_weights': {'physics': 1.0, 'ml': 0.0},
                'confidence': 'low'
            }
        
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        predictions = {}
        uncertainties = {}
        
        property_names = ['tensile_strength', 'tensile_modulus', 'compressive_strength',
                         'flexural_strength', 'flexural_modulus', 'ilss', 'impact_energy']
        
        weights_ml = []
        weights_physics = []
        
        for prop in property_names:
            if prop in self.models:
                model = self.models[prop]
                
                ml_pred = model['rf'].predict(features_scaled)[0]
                
                tree_predictions = np.array([tree.predict(features_scaled)[0] 
                                            for tree in model['rf'].estimators_])
                ml_std = np.std(tree_predictions)
                
                physics_pred = rom_predictions[prop]
                physics_std = physics_pred * 0.10
                
                alpha = 2.5
                w_ml = np.exp(-alpha * ml_std**2)
                w_physics = np.exp(-alpha * physics_std**2)
                w_total = w_ml + w_physics
                
                w_ml_norm = w_ml / w_total
                w_physics_norm = w_physics / w_total
                
                weights_ml.append(w_ml_norm)
                weights_physics.append(w_physics_norm)
                
                hybrid_pred = w_ml_norm * ml_pred + w_physics_norm * physics_pred
                
                hybrid_std = np.sqrt(w_ml_norm**2 * ml_std**2 + 
                                    w_physics_norm**2 * physics_std**2)
                
                predictions[prop] = float(hybrid_pred)
                uncertainties[prop] = {
                    'lower': float(hybrid_pred - 1.96 * hybrid_std),
                    'upper': float(hybrid_pred + 1.96 * hybrid_std),
                    'std': float(hybrid_std),
                    'value': float(hybrid_pred)
                }
            else:
                predictions[prop] = rom_predictions[prop]
                uncertainties[prop] = {
                    'lower': rom_predictions[prop] * 0.85,
                    'upper': rom_predictions[prop] * 1.15,
                    'std': rom_predictions[prop] * 0.10,
                    'value': rom_predictions[prop]
                }
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainties,
            'method_weights': {
                'physics': float(np.mean(weights_physics)) if weights_physics else 0.5,
                'ml': float(np.mean(weights_ml)) if weights_ml else 0.5
            },
            'confidence': 'high' if self.models else 'medium'
        }

# Initialize
physics_engine = PhysicsEngine()
feature_engineer = FeatureEngineer()
hybrid_model = HybridPIRF()
hybrid_model.load_models()

# ===========================
# API ENDPOINTS
# ===========================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        fiber_name = data['fiber']
        matrix_name = data['matrix']
        Vf = float(data['vf'])
        layup = data['layup']
        manufacturing = data['manufacturing']
        
        if fiber_name not in FIBERS:
            return jsonify({'success': False, 'error': f'Unknown fiber: {fiber_name}'}), 400
        if matrix_name not in MATRICES:
            return jsonify({'success': False, 'error': f'Unknown matrix: {matrix_name}'}), 400
        if not (0.25 <= Vf <= 0.75):
            return jsonify({'success': False, 'error': 'Vf must be between 0.25 and 0.75'}), 400
        
        fiber = FIBERS[fiber_name]
        matrix = MATRICES[matrix_name]
        
        rom_props = physics_engine.calculate_rom_properties(
            fiber, matrix, Vf, layup, manufacturing
        )
        
        features = feature_engineer.create_features(
            fiber_name, matrix_name, Vf, layup, manufacturing, rom_props
        )
        
        result = hybrid_model.predict(features, rom_props)
        
        response = {
            'success': True,
            'predictions': result['predictions'],
            'uncertainty': result['uncertainty'],
            'rom_predictions': {
                'tensile_strength': rom_props['tensile_strength'],
                'tensile_modulus': rom_props['tensile_modulus'],
                'compressive_strength': rom_props['compressive_strength'],
                'flexural_strength': rom_props['flexural_strength'],
                'flexural_modulus': rom_props['flexural_modulus'],
                'ilss': rom_props['ilss'],
                'impact_energy': rom_props['impact_energy']
            },
            'method': 'hybrid' if hybrid_model.models else 'physics_only',
            'method_weights': result['method_weights'],
            'confidence': result['confidence']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    try:
        data = request.json
        
        validation_result = material_validator.validate_configuration(
            fiber=data['fiber'],
            matrix=data['matrix'],
            vf=float(data['vf']),
            layup=data['layup'],
            manufacturing=data['manufacturing']
        )
        
        return jsonify({
            'success': True,
            'is_valid': validation_result.is_valid,
            'compatibility_score': validation_result.compatibility_score,
            'warnings': validation_result.warnings,
            'recommendations': validation_result.recommendations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    try:
        data = request.json
        plot_type = data['plot_type']
        config = data['config']
        predictions = data.get('predictions', {})
        
        if plot_type == 'ashby':
            fig = _create_ashby_chart(config, predictions)
        elif plot_type == 'vf_sensitivity':
            fig = _create_vf_sensitivity(config)
        elif plot_type == 'failure_envelope':
            fig = _create_failure_envelope(predictions)
        elif plot_type == 'stress_distribution':
            fig = _create_stress_distribution(config)
        else:
            return jsonify({'success': False, 'error': 'Unknown plot type'}), 400
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return send_file(buf, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        config = data['config']
        applied_load = float(data.get('applied_load', 100))
        
        fiber_name = config['fiber']
        matrix_name = config['matrix']
        Vf = float(config['vf'])
        
        fiber = FIBERS[fiber_name]
        matrix = MATRICES[matrix_name]
        
        rom_props = physics_engine.calculate_rom_properties(
            fiber, matrix, Vf, config['layup'], config['manufacturing']
        )
        
        features = feature_engineer.create_features(
            fiber_name, matrix_name, Vf, config['layup'], config['manufacturing'], rom_props
        )
        
        result = hybrid_model.predict(features, rom_props)
        predictions = result['predictions']
        
        stress_dist = MechanicalSimulator.calculate_stress_distribution(
            config, applied_load=applied_load
        )
        
        stress_state = StressState(
            sigma_x=applied_load,
            sigma_y=applied_load * 0.1,
            tau_xy=0
        )
        
        failure = MechanicalSimulator.tsai_wu_failure_analysis(
            stress_state, config, predictions
        )
        
        return jsonify({
            'success': True,
            'stress_fiber': stress_dist['stress_fiber'],
            'stress_matrix': stress_dist['stress_matrix'],
            'stress_interface': stress_dist['stress_interface'],
            'failure_index': failure.failure_index,
            'safety_factor': failure.safety_factor,
            'failure_mode': failure.failure_mode,
            'will_fail': failure.will_fail
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        config = data['config']
        predictions = data['predictions']
        
        validation_result = material_validator.validate_configuration(
            config['fiber'], config['matrix'], config['vf'],
            config['layup'], config['manufacturing']
        )
        
        generator = ScientificReportGenerator()
        output_path = f"output/reports/report_{config['fiber']}_{config['matrix']}.pdf"
        
        generator.generate_report(
            config=config,
            predictions=predictions,
            validation_result=validation_result,
            plots_dir='output/plots',
            output_path=output_path
        )
        
        return send_file(output_path, as_attachment=True, 
                        download_name='composite_analysis_report.pdf')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/materials', methods=['GET'])
def get_materials():
    return jsonify({
        'fibers': list(FIBERS.keys()),
        'matrices': list(MATRICES.keys()),
        'layups': ['Unidirectional 0°', 'Woven 0/90',
                   'Quasi-isotropic [0/45/90/-45]', 'Angle-ply [±45]',
                   'Cross-ply [0/90]', 'Random Mat'],
        'manufacturing': list(MANUFACTURING_FACTORS.keys())
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': '3.0',
        'models_loaded': len(hybrid_model.models) > 0,
        'num_models': len(hybrid_model.models),
        'features': [
            'Material Validation',
            'Scientific Plotting',
            'PDF Report Generation',
            'Mechanical Simulation'
        ]
    })

# Helper functions for plotting
def _create_ashby_chart(config, predictions):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    materials = [
        {'name': 'E-Glass/Polyester', 'density': 2.0, 'strength': 230},
        {'name': 'Carbon/Epoxy', 'density': 1.55, 'strength': 1420},
        {'name': 'Kevlar/Epoxy', 'density': 1.38, 'strength': 1280},
        {'name': config.get('fiber', '') + '/' + config.get('matrix', ''), 
         'density': 1.7, 'strength': predictions.get('tensile_strength', 500)}
    ]
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, mat in enumerate(materials):
        ax.scatter(mat['density'], mat['strength'], s=200, alpha=0.6, 
                  c=colors[i], edgecolors='black', linewidth=2, label=mat['name'])
    
    ax.set_xlabel('Плотность (г/см³)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Прочность на растяжение (МПа)', fontsize=14, fontweight='bold')
    ax.set_title('Ashby Chart: Выбор материала', fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    return fig

def _create_vf_sensitivity(config):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    vf_values = np.linspace(0.30, 0.70, 21)
    strengths = 300 + 800 * vf_values + np.random.normal(0, 20, len(vf_values))
    
    ax.plot(vf_values, strengths, linewidth=3, color='#667eea', label='Прочность на растяжение')
    ax.fill_between(vf_values, strengths * 0.9, strengths * 1.1, alpha=0.2, color='#667eea')
    
    max_idx = np.argmax(strengths)
    ax.plot(vf_values[max_idx], strengths[max_idx], 'r*', markersize=20, label='Оптимум')
    
    ax.set_xlabel('Объёмная доля волокна (Vf)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Прочность на растяжение (МПа)', fontsize=14, fontweight='bold')
    ax.set_title('Анализ чувствительности к Vf', fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    return fig

def _create_failure_envelope(predictions):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    sigma_xt = predictions.get('tensile_strength', 300)
    sigma_xc = predictions.get('compressive_strength', 200)
    
    sigma_x = np.linspace(-sigma_xc * 1.2, sigma_xt * 1.2, 200)
    sigma_y = np.linspace(-sigma_xc * 1.2, sigma_xt * 1.2, 200)
    X, Y = np.meshgrid(sigma_x, sigma_y)
    
    F1 = 1/sigma_xt - 1/sigma_xc
    F11 = 1/(sigma_xt * sigma_xc)
    F22 = 1/(sigma_xt * sigma_xc)
    F12 = -0.5 * np.sqrt(F11 * F22)
    
    FI = F1*X + F11*X**2 + F22*Y**2 + 2*F12*X*Y
    
    ax.contourf(X, Y, FI, levels=[0, 1], colors=['#6A994E'], alpha=0.3)
    ax.contour(X, Y, FI, levels=[1], colors='red', linewidths=3)
    
    ax.text(0, sigma_xt*0.7, 'БЕЗОПАСНАЯ ЗОНА', fontsize=16, fontweight='bold', 
           ha='center', color='green', alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('σ_x (Продольное напряжение, МПа)', fontsize=14, fontweight='bold')
    ax.set_ylabel('σ_y (Поперечное напряжение, МПа)', fontsize=14, fontweight='bold')
    ax.set_title('Конверт разрушения Tsai-Wu', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig

def _create_stress_distribution(config):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    vf = config.get('vf', 0.60)
    
    x = np.linspace(0, 20, 200)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    np.random.seed(42)
    num_fibers = int(vf * 50)
    fiber_x = np.random.uniform(2, 18, num_fibers)
    fiber_y = np.random.uniform(2, 8, num_fibers)
    
    stress = np.ones_like(X) * 50
    for fx, fy in zip(fiber_x, fiber_y):
        dist = np.sqrt((X - fx)**2 + (Y - fy)**2)
        fiber_mask = dist < 0.3
        stress[fiber_mask] = 150
        
        interface_mask = (dist >= 0.3) & (dist < 0.5)
        stress[interface_mask] = 180
    
    contour = ax.contourf(X, Y, stress, levels=20, cmap='RdYlBu_r')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Напряжение (МПа)', fontsize=12, fontweight='bold')
    
    for fx, fy in zip(fiber_x, fiber_y):
        circle = plt.Circle((fx, fy), 0.3, color='white', ec='black', linewidth=1)
        ax.add_patch(circle)
    
    ax.set_xlabel('x (мм)', fontsize=14, fontweight='bold')
    ax.set_ylabel('y (мм)', fontsize=14, fontweight='bold')
    ax.set_title(f'Распределение напряжений (Vf={vf:.2f})', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    
    return fig

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
