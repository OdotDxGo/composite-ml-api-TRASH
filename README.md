# Composite Materials Property Prediction API

Advanced mechanical property prediction for fiber-reinforced composites using empirical formulas and rule of mixtures.

## Features

- **25 Material Combinations**: 5 fiber types × 5 matrix types
- **7 Mechanical Properties**: Tensile, compressive, flexural strength/modulus, ILSS, impact energy
- **7 Layup Configurations**: UD 0°, UD 90°, Woven, Cross-ply, Quasi-isotropic, etc.
- **7 Manufacturing Processes**: Autoclave, VARTM, RTM, Hand layup, etc.
- **Fast Response**: < 10ms prediction time
- **RESTful API**: JSON-based endpoints
- **Web Interface**: Interactive prediction tool

## Material Database

### Fiber Types
- Carbon
- Glass (E-glass)
- Aramid (Kevlar)
- Basalt
- Natural fibers

### Matrix Types
- Epoxy
- Polyester
- Vinyl ester
- PEEK
- PA6 (Nylon 6)

## API Endpoints

### GET /
Web interface for interactive predictions

### GET /api
API information and documentation

### GET /health
Service health check

### POST /api/predict
Single material prediction

**Request:**
```json
{
  "fiber_type": "Carbon",
  "matrix_type": "Epoxy",
  "fiber_volume_fraction": 0.6,
  "layup": "UD 0°",
  "manufacturing": "Autoclave"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "tensile_strength_MPa": 1500.0,
    "tensile_modulus_GPa": 130.0,
    "compressive_strength_MPa": 1200.0,
    "flexural_strength_MPa": 1400.0,
    "flexural_modulus_GPa": 125.0,
    "ILSS_MPa": 75.0,
    "impact_energy_J": 18.0
  }
}
```

### POST /api/predict/batch
Batch predictions for multiple samples

### GET /api/materials
Complete material database

### GET /api/options
Available options for all parameters

## Prediction Method

**Rule of Mixtures with Correction Factors:**

```
Property = Base_Value × (Vf/0.6) × Layup_Factor × Manufacturing_Factor
```

- Base values: Literature data at Vf = 0.6
- Volume fraction effect: Linear scaling
- Layup factors: 0.4 - 1.0 depending on fiber orientation
- Manufacturing factors: 0.85 - 1.0 depending on process quality

## Technology Stack

- **Backend**: Flask 3.0.0
- **Server**: Gunicorn 21.2.0
- **CORS**: flask-cors 4.0.0
- **Deployment**: Railway.com
- **Method**: Empirical formulas (no ML libraries)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Access at http://localhost:8088
```

## Deployment

Configured for Railway.com with automatic deployment from GitHub.

**Environment Variables:**
- `PORT`: 8088 (set in Railway Variables)

## License

MIT License - See LICENSE file

## Version

2.0 - Production Ready
