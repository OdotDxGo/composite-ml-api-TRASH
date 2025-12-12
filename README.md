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

## Technology Stack

- **Backend**: Flask 3.0.0
- **Server**: Gunicorn 21.2.0
- **CORS**: Flask-CORS 4.0.0
- **Deployment**: Railway.com
- **Method**: Empirical formulas (no ML libraries)

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Access at http://localhost:5000
```

## Deployment

Configured for Railway.com with automatic deployment from GitHub.

Railway automatically provides the PORT environment variable.

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

### POST /api/predict/batch
Batch predictions for multiple samples

### GET /api/materials
Complete material database

### GET /api/options
Available options for all parameters

## License

MIT License

## Version

2.0 - Production Ready
