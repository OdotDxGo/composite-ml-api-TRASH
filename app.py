"""
Enhanced Hybrid Physics-Informed Machine Learning API for Composite Materials
Version: 3.0 (Scientific Edition with Advanced Analysis)
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

# Import new modules
import sys
sys.path.append(os.path.dirname(__file__))

from material_validator import MaterialValidator, ValidationResult
from scientific_plotter import ScientificPlotter
from pdf_report_generator import ScientificReportGenerator
from mechanical_simulator import MechanicalSimulator, StressState

app = Flask(__name__, static_folder='static')
CORS(app)

# Create output directories
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)

# ===========================
# МАТЕРИАЛЬНЫЕ КОНСТАНТЫ
# ===========================

@dataclass
class FiberProperties:
    E: float  # Модуль упругости (GPa)
    sigma: float  # Прочность (MPa)
    rho: float  # Плотность (g/cm³)
    G: float  # Модуль сдвига (GPa)
    epsilon_f: float  # Деформация разрушения (%)
    U_f: float  # Удельная энергия разрушения (J)

@dataclass
class MatrixProperties:
    E: float
    sigma: float
    rho: float
    G: float
    epsilon_f: float
    U_m: float

# Библиотека материалов
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

# Коэффициенты эффективности
EFFICIENCY_FACTORS = {
    'eta_L': 0.95,
    'eta_T': 0.40,
    'eta_strength': 0.90,
    'eta_comp': 0.70,
    'eta_interface': 0.85
}

# Коэффициенты производства
MANUFACTURING_FACTORS = {
    'Autoclave': 1.00,
    'VARTM': 0.93,
    'RTM': 0.88,
    'Compression Molding': 0.85,
    'Hand Layup': 0.75,
    'Filament Winding': 0.90,
    'Pultrusion': 0.92
}

# ===========================
# PHYSICS-BASED CALCULATIONS
# ===========================

class PhysicsEngine:
    """Эмпирические формулы Rule of Mixtures"""
    
    @staticmethod
    def calculate_rom_properties(fiber: FiberProperties, matrix: MatrixProperties, 
                                 Vf: float, layup: str, manufacturing: str) -> Dict[str, float]:
        """Расчёт свойств по правилу смесей"""
        
        eta_L = EFFICIENCY_FACTORS['eta_L']
        eta_T = EFFICIENCY_FACTORS['eta_T']
        eta_strength = EFFICIENCY_FACTORS['eta_strength']
        eta_comp = EFFICIENCY_FACTORS['eta_comp']
        eta_interface = EFFICIENCY_FACTORS['eta_interface']
        eta_mfg = MANUFACTURING_FACTORS[manufacturing]
        
        # Unidirectional properties
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
        
        # Layup corrections
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
        
        # Manufacturing corrections
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

# ===========================
# FEATURE ENGINEERING
# ===========================

class FeatureEngineer:
    """Создание physics-informed features для ML"""
    
    @staticmethod
    def create_features(fiber_name: str, matrix_name: str, Vf: float, 
                       layup: str, manufacturing: str, 
                       rom_properties: Dict[str, float]) -> np.ndarray:
        """Создаёт 26 features для ML модели"""
        
        features = []
        
        # Base features (5)
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
        
        # Physics-derived features (7)
        features.extend([
            rom_properties['tensile_strength'],
            rom_properties['tensile_modulus'],
            rom_properties['compressive_strength'],
            rom_properties['flexural_strength'],
            rom_properties['flexural_modulus'],
            rom_properties['ilss'],
            rom_properties['impact_energy']
        ])
        
        # Constituent property ratios (4)
        features.extend([
            rom_properties['E_f_E_m_ratio'],
            rom_properties['sigma_f_sigma_m_ratio'],
            rom_properties['rho_f_rho_m_ratio'],
            rom_properties['G_f_G_m_ratio']
        ])
        
        # Volume fraction transformations (3)
        features.extend([
            Vf ** 2,
            Vf ** 3,
            1 / (1 - Vf + 1e-6)
        ])
        
        # Interaction terms (7)
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

# ===========================
# HYBRID ML MODEL
# ===========================

class HybridPIRF:
    """Physics-Informed Random Forest - Гибридная модель"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
    def load_models(self):
        """Загрузка обученных моделей"""
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
        else:
            print("⚠ No scaler found.")
    
    def predict(self, features: np.ndarray, rom_predictions: Dict[str, float]) -> Dict[str, any]:
        """Гибридное предсказание с uncertainty quantification"""
        
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

# ===========================
# INITIALIZE
# ===========================

physics_engine = PhysicsEngine()
feature_engineer = FeatureEngineer()
hybrid_model = HybridPIRF()
material_validator = MaterialValidator()
scientific_plotter = ScientificPlotter()

hybrid_model.load_models()

# ===========================
# API ENDPOINTS
# ===========================

@app.route('/')
def index():
    """Serve frontend"""
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
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
            'confidence': result['confidence'],
            'inputs': {
                'fiber': fiber_name,
                'matrix': matrix_name,
                'vf': Vf,
                'layup': layup,
                'manufacturing': manufacturing
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/validate', methods=['POST'])
def validate():
    """Material validation endpoint"""
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
            'recommendations': validation_result.recommendations,
            'alternative_configs': validation_result.alternative_configs
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate PDF scientific report"""
    try:
        data = request.json
        
        config = data['config']
        predictions = data['predictions']
        
        # Get validation
        validation_result = material_validator.validate_configuration(
            config['fiber'], config['matrix'], config['vf'],
            config['layup'], config['manufacturing']
        )
        
        # Generate report
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
    """Get available materials"""
    return jsonify({
        'fibers': list(FIBERS.keys()),
        'matrices': list(MATRICES.keys()),
        'layups': ['Unidirectional 0°', 'Unidirectional 90°', 'Woven 0/90',
                   'Quasi-isotropic [0/45/90/-45]', 'Angle-ply [±45]',
                   'Cross-ply [0/90]', 'Random Mat'],
        'manufacturing': list(MANUFACTURING_FACTORS.keys())
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
