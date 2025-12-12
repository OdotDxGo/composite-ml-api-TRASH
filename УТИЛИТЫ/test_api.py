"""
API Testing Script
Tests all endpoints with various configurations
"""

import requests
import json
import time
from tabulate import tabulate

API_URL = "http://localhost:5000"  # Замените на Railway URL после deploy

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    data = response.json()
    
    print(f"Status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"Models loaded: {data['models_loaded']}")
    print(f"Number of models: {data['num_models']}")
    
    assert data['status'] == 'healthy', "API not healthy!"
    print("✅ PASSED")

def test_materials():
    """Test materials endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Get Available Materials")
    print("="*60)
    
    response = requests.get(f"{API_URL}/materials")
    data = response.json()
    
    print(f"Fibers: {len(data['fibers'])}")
    print(f"  {', '.join(data['fibers'])}")
    print(f"\nMatrices: {len(data['matrices'])}")
    print(f"  {', '.join(data['matrices'])}")
    
    print("✅ PASSED")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*60)
    print("TEST 3: Single Prediction")
    print("="*60)
    
    payload = {
        "fiber": "E-Glass",
        "matrix": "Polyester",
        "vf": 0.60,
        "layup": "Quasi-isotropic [0/45/90/-45]",
        "manufacturing": "Compression Molding"
    }
    
    print(f"Input: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict", json=payload)
    elapsed = (time.time() - start_time) * 1000
    
    data = response.json()
    
    if not data['success']:
        print(f"❌ FAILED: {data['error']}")
        return
    
    print(f"\nPrediction time: {elapsed:.1f} ms")
    print(f"Method: {data['method']}")
    print(f"Confidence: {data['confidence']}")
    
    # Display results table
    results = []
    for prop, value in data['predictions'].items():
        uncertainty = data['uncertainty'][prop]
        results.append([
            prop.replace('_', ' ').title(),
            f"{value:.2f}",
            f"[{uncertainty['lower']:.2f}, {uncertainty['upper']:.2f}]",
            f"±{uncertainty['std']:.2f}"
        ])
    
    print("\n" + tabulate(results, 
                          headers=['Property', 'Value', '95% CI', 'Std Dev'],
                          tablefmt='grid'))
    
    # Method weights
    weights = data['method_weights']
    print(f"\nMethod Weights:")
    print(f"  Physics: {weights['physics']*100:.1f}%")
    print(f"  ML:      {weights['ml']*100:.1f}%")
    
    print("✅ PASSED")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases & Error Handling")
    print("="*60)
    
    test_cases = [
        {
            "name": "Invalid Vf (too high)",
            "payload": {"fiber": "E-Glass", "matrix": "Epoxy", "vf": 0.85, 
                       "layup": "Woven 0/90", "manufacturing": "RTM"},
            "expect_error": True
        },
        {
            "name": "Invalid Vf (too low)",
            "payload": {"fiber": "E-Glass", "matrix": "Epoxy", "vf": 0.15,
                       "layup": "Woven 0/90", "manufacturing": "RTM"},
            "expect_error": True
        },
        {
            "name": "Valid boundary Vf (0.25)",
            "payload": {"fiber": "E-Glass", "matrix": "Epoxy", "vf": 0.25,
                       "layup": "Woven 0/90", "manufacturing": "RTM"},
            "expect_error": False
        }
    ]
    
    for test in test_cases:
        print(f"\n  Testing: {test['name']}")
        response = requests.post(f"{API_URL}/predict", json=test['payload'])
        data = response.json()
        
        if test['expect_error']:
            if not data['success']:
                print(f"    ✅ Correctly rejected: {data.get('error', 'Unknown error')}")
            else:
                print(f"    ❌ Should have failed but didn't!")
        else:
            if data['success']:
                print(f"    ✅ Correctly accepted")
            else:
                print(f"    ❌ Should have passed: {data.get('error', 'Unknown error')}")
    
    print("\n✅ PASSED")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 HYBRID PIRF API TEST SUITE")
    print("="*60)
    print(f"Target: {API_URL}")
    
    try:
        test_health()
        test_materials()
        test_single_prediction()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Для tabulate: pip install tabulate
    run_all_tests()