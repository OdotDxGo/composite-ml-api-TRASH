"""
Scientific Publication-Quality Plotting System
For Scopus Q1-Q2 journal papers and doctoral dissertations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ScientificPlotter:
    """
    Creates publication-ready plots for composite materials research
    All plots are 300 DPI, vector-compatible, with proper fonts
    """
    
    # Publication settings
    FIGSIZE_SINGLE = (8, 6)
    FIGSIZE_DOUBLE = (12, 5)
    FIGSIZE_GRID = (14, 10)
    DPI = 300
    FONT_SIZE = 12
    FONT_FAMILY = 'serif'
    
    @staticmethod
    def plot_ashby_chart(data: List[Dict], save_path: str = None):
        """
        Ashby Material Selection Chart
        X-axis: Density (g/cm³)
        Y-axis: Tensile Strength (MPa)
        
        Perfect for material selection in papers
        """
        
        fig, ax = plt.subplots(figsize=ScientificPlotter.FIGSIZE_SINGLE, dpi=ScientificPlotter.DPI)
        
        # Extract data
        fibers = [d['fiber'] for d in data]
        matrices = [d['matrix'] for d in data]
        densities = [d['density'] for d in data]
        strengths = [d['tensile_strength'] for d in data]
        
        # Color by fiber type
        fiber_colors = {
            'E-Glass': '#2E86AB',
            'Carbon T300': '#A23B72',
            'Kevlar 49': '#F18F01',
            'Basalt': '#C73E1D',
            'Flax': '#6A994E'
        }
        
        colors = [fiber_colors.get(f, '#999999') for f in fibers]
        
        # Scatter plot
        scatter = ax.scatter(densities, strengths, c=colors, s=150, alpha=0.7,
                            edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, (fiber, matrix, x, y) in enumerate(zip(fibers, matrices, densities, strengths)):
            label = f"{fiber[:10]}/\n{matrix[:8]}"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Add performance indices (constant lines)
        x_range = np.linspace(min(densities)*0.9, max(densities)*1.1, 100)
        
        # Specific strength lines (σ/ρ = constant)
        for spec_strength in [100, 200, 300, 400]:
            y_line = spec_strength * x_range
            ax.plot(x_range, y_line, '--', color='gray', alpha=0.3, linewidth=0.8)
            ax.text(x_range[-1]*1.01, y_line[-1], f'{spec_strength}', 
                   fontsize=8, color='gray', alpha=0.5)
        
        ax.set_xlabel('Density (g/cm³)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Tensile Strength (MPa)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_title('Ashby Chart: Composite Material Selection', 
                    fontsize=ScientificPlotter.FONT_SIZE+2, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        legend_elements = [plt.scatter([], [], c=color, s=150, alpha=0.7, edgecolors='black', linewidth=1.5, label=fiber)
                          for fiber, color in fiber_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_vf_sensitivity(vf_values: np.ndarray, properties: Dict[str, np.ndarray], 
                           save_path: str = None):
        """
        Volume Fraction Sensitivity Analysis
        Shows how properties change with Vf
        
        Perfect for optimization studies
        """
        
        fig, axes = plt.subplots(2, 2, figsize=ScientificPlotter.FIGSIZE_GRID, dpi=ScientificPlotter.DPI)
        axes = axes.flatten()
        
        # Define properties to plot
        props_to_plot = [
            ('tensile_strength', 'Tensile Strength (MPa)', '#2E86AB'),
            ('tensile_modulus', 'Tensile Modulus (GPa)', '#A23B72'),
            ('ilss', 'ILSS (MPa)', '#F18F01'),
            ('impact_energy', 'Impact Energy (J)', '#6A994E')
        ]
        
        for idx, (prop_key, prop_label, color) in enumerate(props_to_plot):
            ax = axes[idx]
            
            if prop_key in properties:
                values = properties[prop_key]
                
                # Main line
                ax.plot(vf_values, values, linewidth=3, color=color, label='Prediction')
                
                # Confidence interval (if available)
                if f'{prop_key}_lower' in properties:
                    lower = properties[f'{prop_key}_lower']
                    upper = properties[f'{prop_key}_upper']
                    ax.fill_between(vf_values, lower, upper, alpha=0.2, color=color, label='95% CI')
                
                # Find optimal point
                max_idx = np.argmax(values)
                ax.plot(vf_values[max_idx], values[max_idx], 'r*', markersize=20, 
                       label=f'Optimal: Vf={vf_values[max_idx]:.2f}')
                
                # Annotations
                ax.axvline(vf_values[max_idx], color='red', linestyle='--', alpha=0.3)
                ax.text(vf_values[max_idx], values[max_idx]*0.95, 
                       f'Max: {values[max_idx]:.1f}', 
                       ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Volume Fraction (Vf)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
                ax.set_ylabel(prop_label, fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
                ax.set_title(f'{prop_label.split("(")[0].strip()} vs Vf', 
                            fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
        
        plt.suptitle('Volume Fraction Sensitivity Analysis', 
                    fontsize=ScientificPlotter.FONT_SIZE+4, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_radar_comparison(configs: List[Dict], properties: List[str], 
                             config_names: List[str], save_path: str = None):
        """
        Radar/Spider Chart for Multi-Config Comparison
        Perfect for comparing different designs
        """
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=ScientificPlotter.DPI, 
                              subplot_kw=dict(projection='polar'))
        
        # Number of properties
        N = len(properties)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each configuration
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
        
        for idx, (config, name) in enumerate(zip(configs, config_names)):
            values = [config[prop] for prop in properties]
            
            # Normalize to 0-100 scale
            max_vals = [max([c[prop] for c in configs]) for prop in properties]
            normalized = [(v / mv * 100) if mv > 0 else 0 for v, mv in zip(values, max_vals)]
            normalized += normalized[:1]  # Complete the circle
            
            ax.plot(angles, normalized, 'o-', linewidth=2, color=colors[idx % len(colors)], 
                   label=name, markersize=8)
            ax.fill(angles, normalized, alpha=0.15, color=colors[idx % len(colors)])
        
        # Set labels
        property_labels = [p.replace('_', ' ').title() for p in properties]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(property_labels, fontsize=ScientificPlotter.FONT_SIZE)
        
        # Set y-axis
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        ax.set_title('Multi-Configuration Comparison\n(Normalized Properties)', 
                    fontsize=ScientificPlotter.FONT_SIZE+2, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_failure_envelope(config: Dict, save_path: str = None):
        """
        Tsai-Wu Failure Envelope
        Shows safe operating region for composite
        
        Essential for mechanical design
        """
        
        fig, ax = plt.subplots(figsize=ScientificPlotter.FIGSIZE_SINGLE, dpi=ScientificPlotter.DPI)
        
        # Extract properties
        sigma_xt = config.get('tensile_strength', 300)  # Tension
        sigma_xc = config.get('compressive_strength', 200)  # Compression
        
        # Create grid
        sigma_x = np.linspace(-sigma_xc*1.2, sigma_xt*1.2, 200)
        sigma_y = np.linspace(-sigma_xc*1.2, sigma_xt*1.2, 200)
        X, Y = np.meshgrid(sigma_x, sigma_y)
        
        # Tsai-Wu failure criterion (simplified 2D)
        F1 = 1/sigma_xt - 1/sigma_xc
        F11 = 1/(sigma_xt * sigma_xc)
        F22 = 1/(sigma_xt * sigma_xc)
        F12 = -0.5 * np.sqrt(F11 * F22)  # Interaction term
        
        failure_index = F1*X + F11*X**2 + F22*Y**2 + 2*F12*X*Y
        
        # Plot failure envelope
        contour = ax.contourf(X, Y, failure_index, levels=[0, 1], colors=['#6A994E'], alpha=0.3)
        ax.contour(X, Y, failure_index, levels=[1], colors='red', linewidths=3)
        
        # Add safe/fail regions
        ax.text(0, sigma_xt*0.7, 'SAFE ZONE', fontsize=16, fontweight='bold', 
               ha='center', color='green', alpha=0.7)
        ax.text(sigma_xt*0.8, sigma_xt*0.8, 'FAILURE', fontsize=14, fontweight='bold', 
               ha='center', color='red', alpha=0.7, rotation=45)
        
        # Add axes
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
        
        # Labels
        ax.set_xlabel('σ_x (Longitudinal Stress, MPa)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_ylabel('σ_y (Transverse Stress, MPa)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_title('Tsai-Wu Failure Envelope\n(First Ply Failure Criterion)', 
                    fontsize=ScientificPlotter.FONT_SIZE+2, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_uncertainty_analysis(predictions: Dict, save_path: str = None):
        """
        Uncertainty Quantification Visualization
        Shows prediction confidence intervals
        
        Critical for ML validation in papers
        """
        
        fig, ax = plt.subplots(figsize=ScientificPlotter.FIGSIZE_DOUBLE, dpi=ScientificPlotter.DPI)
        
        properties = list(predictions.keys())
        x_pos = np.arange(len(properties))
        
        values = [predictions[p]['value'] for p in properties]
        lower = [predictions[p]['lower'] for p in properties]
        upper = [predictions[p]['upper'] for p in properties]
        errors = [[values[i] - lower[i], upper[i] - values[i]] for i in range(len(values))]
        
        # Normalize to show relative uncertainty
        colors = []
        for i, p in enumerate(properties):
            rel_error = (upper[i] - lower[i]) / values[i]
            if rel_error < 0.1:
                colors.append('#6A994E')  # Low uncertainty
            elif rel_error < 0.2:
                colors.append('#F18F01')  # Medium
            else:
                colors.append('#C73E1D')  # High uncertainty
        
        # Bar plot with error bars
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(x_pos, values, yerr=np.array(errors).T, fmt='none', 
                   ecolor='black', capsize=5, capthick=2)
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in properties], 
                          rotation=45, ha='right', fontsize=ScientificPlotter.FONT_SIZE-1)
        ax.set_ylabel('Property Value', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_title('Prediction Uncertainty Analysis (95% Confidence Intervals)', 
                    fontsize=ScientificPlotter.FONT_SIZE+2, fontweight='bold', pad=20)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#6A994E', edgecolor='black', label='Low Uncertainty (<10%)'),
            Patch(facecolor='#F18F01', edgecolor='black', label='Medium (10-20%)'),
            Patch(facecolor='#C73E1D', edgecolor='black', label='High (>20%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_stress_distribution(width: int = 20, height: int = 10, vf: float = 0.60,
                                 stress_applied: float = 100, save_path: str = None):
        """
        2D Stress Distribution Visualization
        Simulates stress in fiber-matrix system
        
        Beautiful for presentations and papers
        """
        
        fig, ax = plt.subplots(figsize=ScientificPlotter.FIGSIZE_DOUBLE, dpi=ScientificPlotter.DPI)
        
        # Create mesh
        x = np.linspace(0, width, 200)
        y = np.linspace(0, height, 100)
        X, Y = np.meshgrid(x, y)
        
        # Simulate fiber locations
        np.random.seed(42)
        num_fibers = int(vf * 50)
        fiber_x = np.random.uniform(2, width-2, num_fibers)
        fiber_y = np.random.uniform(2, height-2, num_fibers)
        fiber_r = 0.3
        
        # Calculate stress distribution
        stress = np.ones_like(X) * stress_applied * 0.5  # Matrix stress
        
        for fx, fy in zip(fiber_x, fiber_y):
            dist = np.sqrt((X - fx)**2 + (Y - fy)**2)
            # Higher stress in fibers
            fiber_mask = dist < fiber_r
            stress[fiber_mask] = stress_applied * 1.5
            
            # Stress concentration at interface
            interface_mask = (dist >= fiber_r) & (dist < fiber_r + 0.2)
            stress[interface_mask] = stress_applied * 1.8
        
        # Plot stress contour
        contour = ax.contourf(X, Y, stress, levels=20, cmap='RdYlBu_r')
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Stress (MPa)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        
        # Draw fibers
        for fx, fy in zip(fiber_x, fiber_y):
            circle = plt.Circle((fx, fy), fiber_r, color='white', ec='black', linewidth=1)
            ax.add_patch(circle)
        
        # Applied load arrows
        arrow_y = height * 0.5
        ax.annotate('', xy=(width*0.95, arrow_y), xytext=(width*0.85, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='red'))
        ax.text(width*0.98, arrow_y, 'Load', fontsize=12, color='red', fontweight='bold', va='center')
        
        ax.set_xlabel('x (mm)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_ylabel('y (mm)', fontsize=ScientificPlotter.FONT_SIZE, fontweight='bold')
        ax.set_title(f'Stress Distribution in Composite (Vf={vf:.2f})', 
                    fontsize=ScientificPlotter.FONT_SIZE+2, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=ScientificPlotter.DPI, bbox_inches='tight')
        
        return fig


# Example usage
if __name__ == "__main__":
    plotter = ScientificPlotter()
    
    # Example data
    sample_data = [
        {'fiber': 'E-Glass', 'matrix': 'Polyester', 'density': 2.0, 'tensile_strength': 230},
        {'fiber': 'Carbon T300', 'matrix': 'Epoxy', 'density': 1.55, 'tensile_strength': 1420},
        {'fiber': 'Kevlar 49', 'matrix': 'Epoxy', 'density': 1.38, 'tensile_strength': 1280},
        {'fiber': 'Basalt', 'matrix': 'Vinyl Ester', 'density': 2.1, 'tensile_strength': 480},
    ]
    
    print("Generating publication-quality plots...")
    
    # Ashby chart
    plotter.plot_ashby_chart(sample_data, '/tmp/ashby_chart.png')
    print("✓ Ashby chart saved")
    
    # Stress distribution
    plotter.plot_stress_distribution(save_path='/tmp/stress_distribution.png')
    print("✓ Stress distribution saved")
    
    print("\nAll plots generated at 300 DPI, ready for publication!")
