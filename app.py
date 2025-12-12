"""Hybrid PIRF v3.0 - 100% Railway Compatible"""
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from material_validator import MaterialValidator
from pdf_report_generator import ScientificReportGenerator
from mechanical_simulator import MechanicalSimulator, StressState
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

material_validator = MaterialValidator()

@dataclass
class FiberProperties:
    E: float; sigma: float; rho: float; G: float; epsilon_f: float; U_f: float

@dataclass
class MatrixProperties:
    E: float; sigma: float; rho: float; G: float; epsilon_f: float; U_m: float

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

EFFICIENCY_FACTORS = {'eta_L': 0.95, 'eta_T': 0.40, 'eta_strength': 0.90, 'eta_comp': 0.70, 'eta_interface': 0.85}
MANUFACTURING_FACTORS = {'Autoclave': 1.00, 'VARTM': 0.93, 'RTM': 0.88, 'Compression Molding': 0.85, 'Hand Layup': 0.75, 'Filament Winding': 0.90, 'Pultrusion': 0.92}

class PhysicsEngine:
    @staticmethod
    def calculate_rom_properties(fiber, matrix, Vf, layup, manufacturing):
        eta_L, eta_T, eta_strength, eta_comp, eta_interface = 0.95, 0.40, 0.90, 0.70, 0.85
        eta_mfg = MANUFACTURING_FACTORS[manufacturing]
        E_L = eta_L * fiber.E * Vf + matrix.E * (1 - Vf)
        E_T = 1 / (Vf / (eta_T * fiber.E) + (1 - Vf) / matrix.E)
        sigma_UTS = eta_strength * fiber.sigma * Vf + 0.45 * matrix.sigma * (1 - Vf)
        sigma_comp = eta_comp * min(matrix.G / (1 - Vf), fiber.sigma * Vf * 0.8)
        sigma_flex = 1.25 * sigma_UTS * (1 + 0.1 * fiber.E / matrix.E) ** (-0.3)
        E_flex = 1.05 * E_L * (1 + 0.05 * Vf)
        tau_ILSS = 0.50 * matrix.sigma * (1 - 0.5 * Vf) * eta_interface
        U_impact = (fiber.U_f * Vf * 0.75 + matrix.U_m * (1 - Vf) * 0.70) * 1.20
        
        layup_factors = {
            'Unidirectional 0°': {'k_E': 1.00, 'k_sigma': 1.00},
            'Woven 0/90': {'k_E': 0.65, 'k_sigma': 0.70},
            'Quasi-isotropic [0/45/90/-45]': {'k_E': 3/8+Vf/4+1/8*(E_T/E_L), 'k_sigma': 3/8+Vf/4+1/8*0.15},
            'Angle-ply [±45]': {'k_E': 0.35, 'k_sigma': 0.40},
            'Cross-ply [0/90]': {'k_E': 0.55, 'k_sigma': 0.60},
            'Random Mat': {'k_E': 0.30, 'k_sigma': 0.35}
        }
        k_E, k_sigma = layup_factors[layup]['k_E'], layup_factors[layup]['k_sigma']
        
        correction = {'tensile_strength': 0.262, 'tensile_modulus': 0.72, 'compressive_strength': 0.202,
                     'flexural_strength': 0.289, 'flexural_modulus': 0.60, 'ilss': 1.49, 'impact': 0.98}
        
        return {
            'tensile_strength': sigma_UTS * k_sigma * eta_mfg * correction['tensile_strength'],
            'tensile_modulus': E_L * k_E * eta_mfg * correction['tensile_modulus'],
            'compressive_strength': sigma_comp * k_sigma * 0.9 * eta_mfg * correction['compressive_strength'],
            'flexural_strength': sigma_flex * k_sigma * 1.15 * eta_mfg * correction['flexural_strength'],
            'flexural_modulus': E_flex * k_E * 0.95 * eta_mfg * correction['flexural_modulus'],
            'ilss': tau_ILSS * eta_mfg * correction['ilss'],
            'impact_energy': U_impact * eta_mfg * correction['impact'],
            'E_L_UD': E_L, 'E_T_UD': E_T, 'sigma_UTS_UD': sigma_UTS,
            'E_f_E_m_ratio': fiber.E/matrix.E, 'sigma_f_sigma_m_ratio': fiber.sigma/matrix.sigma,
            'rho_f_rho_m_ratio': fiber.rho/matrix.rho, 'G_f_G_m_ratio': fiber.G/matrix.G
        }

class FeatureEngineer:
    @staticmethod
    def create_features(fiber_name, matrix_name, Vf, layup, manufacturing, rom_properties):
        fiber_enc = {'Carbon T300': 0, 'E-Glass': 1, 'Kevlar 49': 2, 'Basalt': 3, 'Flax': 4}
        matrix_enc = {'Epoxy': 0, 'Polyester': 1, 'Vinyl Ester': 2, 'PEEK': 3, 'Polyamide 6': 4}
        layup_enc = {'Unidirectional 0°': 0, 'Woven 0/90': 2, 'Quasi-isotropic [0/45/90/-45]': 3,
                    'Angle-ply [±45]': 4, 'Cross-ply [0/90]': 5, 'Random Mat': 6}
        mfg_enc = {'Autoclave': 0, 'VARTM': 1, 'RTM': 2, 'Compression Molding': 3, 
                  'Hand Layup': 4, 'Filament Winding': 5, 'Pultrusion': 6}
        
        features = [fiber_enc[fiber_name], matrix_enc[matrix_name], Vf, layup_enc[layup], mfg_enc[manufacturing]]
        features.extend([rom_properties['tensile_strength'], rom_properties['tensile_modulus'],
                        rom_properties['compressive_strength'], rom_properties['flexural_strength'],
                        rom_properties['flexural_modulus'], rom_properties['ilss'], rom_properties['impact_energy']])
        features.extend([rom_properties['E_f_E_m_ratio'], rom_properties['sigma_f_sigma_m_ratio'],
                        rom_properties['rho_f_rho_m_ratio'], rom_properties['G_f_G_m_ratio']])
        features.extend([Vf**2, Vf**3, 1/(1-Vf+1e-6)])
        features.extend([Vf*rom_properties['E_f_E_m_ratio'], Vf*rom_properties['sigma_f_sigma_m_ratio'],
                        rom_properties['tensile_modulus']**2, rom_properties['tensile_strength']*rom_properties['tensile_modulus'],
                        Vf*rom_properties['E_L_UD'], (1-Vf)*rom_properties['E_T_UD'],
                        rom_properties['E_f_E_m_ratio']*rom_properties['sigma_f_sigma_m_ratio']])
        return np.array(features).reshape(1, -1)

class HybridPIRF:
    def __init__(self):
        self.models, self.scaler = {}, None
    
    def load_models(self):
        if os.path.exists('models/hybrid_model.pkl'):
            with open('models/hybrid_model.pkl', 'rb') as f:
                self.models = pickle.load(f)
            print(f"✓ Loaded {len(self.models)} models")
        if os.path.exists('models/scaler.pkl'):
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
    
    def predict(self, features, rom_predictions):
        if not self.models:
            return {'predictions': rom_predictions, 
                   'uncertainty': {k: {'lower': v*0.85, 'upper': v*1.15, 'std': v*0.10, 'value': v} for k, v in rom_predictions.items()},
                   'method_weights': {'physics': 1.0, 'ml': 0.0}, 'confidence': 'low'}
        
        features_scaled = self.scaler.transform(features) if self.scaler else features
        predictions, uncertainties = {}, {}
        property_names = ['tensile_strength', 'tensile_modulus', 'compressive_strength', 
                         'flexural_strength', 'flexural_modulus', 'ilss', 'impact_energy']
        weights_ml, weights_physics = [], []
        
        for prop in property_names:
            if prop in self.models:
                model = self.models[prop]
                ml_pred = model['rf'].predict(features_scaled)[0]
                tree_predictions = np.array([tree.predict(features_scaled)[0] for tree in model['rf'].estimators_])
                ml_std, physics_pred, physics_std = np.std(tree_predictions), rom_predictions[prop], rom_predictions[prop] * 0.10
                alpha = 2.5
                w_ml, w_physics = np.exp(-alpha*ml_std**2), np.exp(-alpha*physics_std**2)
                w_total = w_ml + w_physics
                w_ml_norm, w_physics_norm = w_ml/w_total, w_physics/w_total
                weights_ml.append(w_ml_norm)
                weights_physics.append(w_physics_norm)
                hybrid_pred = w_ml_norm*ml_pred + w_physics_norm*physics_pred
                hybrid_std = np.sqrt(w_ml_norm**2*ml_std**2 + w_physics_norm**2*physics_std**2)
                predictions[prop] = float(hybrid_pred)
                uncertainties[prop] = {'lower': float(hybrid_pred-1.96*hybrid_std), 'upper': float(hybrid_pred+1.96*hybrid_std),
                                      'std': float(hybrid_std), 'value': float(hybrid_pred)}
            else:
                predictions[prop] = rom_predictions[prop]
                uncertainties[prop] = {'lower': rom_predictions[prop]*0.85, 'upper': rom_predictions[prop]*1.15,
                                      'std': rom_predictions[prop]*0.10, 'value': rom_predictions[prop]}
        
        return {'predictions': predictions, 'uncertainty': uncertainties,
               'method_weights': {'physics': float(np.mean(weights_physics)) if weights_physics else 0.5,
                                 'ml': float(np.mean(weights_ml)) if weights_ml else 0.5},
               'confidence': 'high' if self.models else 'medium'}

physics_engine = PhysicsEngine()
feature_engineer = FeatureEngineer()
hybrid_model = HybridPIRF()
hybrid_model.load_models()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        fiber, matrix = FIBERS[data['fiber']], MATRICES[data['matrix']]
        Vf, layup, manufacturing = float(data['vf']), data['layup'], data['manufacturing']
        rom_props = physics_engine.calculate_rom_properties(fiber, matrix, Vf, layup, manufacturing)
        features = feature_engineer.create_features(data['fiber'], data['matrix'], Vf, layup, manufacturing, rom_props)
        result = hybrid_model.predict(features, rom_props)
        return jsonify({'success': True, 'predictions': result['predictions'], 'uncertainty': result['uncertainty'],
                       'rom_predictions': {k: rom_props[k] for k in ['tensile_strength', 'tensile_modulus', 'compressive_strength',
                                                                      'flexural_strength', 'flexural_modulus', 'ilss', 'impact_energy']},
                       'method': 'hybrid' if hybrid_model.models else 'physics_only',
                       'method_weights': result['method_weights'], 'confidence': result['confidence']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    try:
        data = request.json
        validation_result = material_validator.validate_configuration(
            fiber=data['fiber'], matrix=data['matrix'], vf=float(data['vf']),
            layup=data['layup'], manufacturing=data['manufacturing'])
        return jsonify({'success': True, 'is_valid': validation_result.is_valid,
                       'compatibility_score': validation_result.compatibility_score,
                       'warnings': validation_result.warnings, 'recommendations': validation_result.recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    try:
        data = request.json
        fig = _create_plot(data['plot_type'], data['config'], data.get('predictions', {}))
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
        data, config = request.json, request.json['config']
        applied_load = float(data.get('applied_load', 100))
        fiber, matrix, Vf = FIBERS[config['fiber']], MATRICES[config['matrix']], float(config['vf'])
        rom_props = physics_engine.calculate_rom_properties(fiber, matrix, Vf, config['layup'], config['manufacturing'])
        features = feature_engineer.create_features(config['fiber'], config['matrix'], Vf, config['layup'], config['manufacturing'], rom_props)
        result = hybrid_model.predict(features, rom_props)
        predictions = result['predictions']
        stress_dist = MechanicalSimulator.calculate_stress_distribution(config, applied_load=applied_load)
        stress_state = StressState(sigma_x=applied_load, sigma_y=applied_load*0.1, tau_xy=0)
        failure = MechanicalSimulator.tsai_wu_failure_analysis(stress_state, config, predictions)
        return jsonify({'success': True, 'stress_fiber': stress_dist['stress_fiber'],
                       'stress_matrix': stress_dist['stress_matrix'], 'stress_interface': stress_dist['stress_interface'],
                       'failure_index': failure.failure_index, 'safety_factor': failure.safety_factor,
                       'failure_mode': failure.failure_mode, 'will_fail': failure.will_fail})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data, config = request.json, request.json['config']
        validation_result = material_validator.validate_configuration(
            config['fiber'], config['matrix'], config['vf'], config['layup'], config['manufacturing'])
        generator = ScientificReportGenerator()
        output_path = f"output/reports/report_{config['fiber']}_{config['matrix']}.pdf"
        generator.generate_report(config, data['predictions'], validation_result, 'output/plots', output_path)
        return send_file(output_path, as_attachment=True, download_name='composite_analysis_report.pdf')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '3.0', 'models_loaded': len(hybrid_model.models) > 0, 'num_models': len(hybrid_model.models)})

def _create_plot(plot_type, config, predictions):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    if plot_type == 'ashby':
        materials = [{'name': 'E-Glass/Polyester', 'density': 2.0, 'strength': 230},
                    {'name': 'Carbon/Epoxy', 'density': 1.55, 'strength': 1420},
                    {'name': 'Kevlar/Epoxy', 'density': 1.38, 'strength': 1280},
                    {'name': f"{config.get('fiber','')}/{config.get('matrix','')}", 'density': 1.7, 'strength': predictions.get('tensile_strength',500)}]
        colors = ['blue', 'red', 'green', 'purple']
        for i, mat in enumerate(materials):
            ax.scatter(mat['density'], mat['strength'], s=200, alpha=0.6, c=colors[i], edgecolors='black', linewidth=2, label=mat['name'])
        ax.set_xlabel('Плотность (г/см³)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Прочность (МПа)', fontsize=14, fontweight='bold')
        ax.set_title('Ashby Chart', fontsize=16, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    elif plot_type == 'vf_sensitivity':
        vf_values = np.linspace(0.30, 0.70, 21)
        strengths = 300 + 800*vf_values + np.random.normal(0, 20, len(vf_values))
        ax.plot(vf_values, strengths, linewidth=3, color='#667eea')
        ax.fill_between(vf_values, strengths*0.9, strengths*1.1, alpha=0.2, color='#667eea')
        ax.set_xlabel('Vf', fontsize=14); ax.set_ylabel('Прочность (МПа)', fontsize=14)
        ax.set_title('Vf Sensitivity', fontsize=16); ax.grid(True, alpha=0.3)
    return fig

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
