"""
API Monitoring and Health Check System
Continuous monitoring with logging
"""

import requests
import time
from datetime import datetime

API_URL = "http://localhost:5000"

def health_check():
    """Perform health check"""
    try:
        start_time = time.time()
        response = requests.get(f"{API_URL}/health", timeout=5)
        elapsed = (time.time() - start_time) * 1000  # ms
        
        data = response.json()
        
        return {
            'timestamp': datetime.now(),
            'status': 'up' if response.status_code == 200 else 'down',
            'response_time_ms': elapsed,
            'models_loaded': data.get('models_loaded', False),
            'version': data.get('version', 'unknown')
        }
    except Exception as e:
        return {
            'timestamp': datetime.now(),
            'status': 'down',
            'error': str(e)
        }

def prediction_test():
    """Test prediction endpoint"""
    test_config = {
        'fiber': 'E-Glass',
        'matrix': 'Polyester',
        'vf': 0.60,
        'layup': 'Woven 0/90',
        'manufacturing': 'VARTM'
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_config, timeout=10)
        elapsed = (time.time() - start_time) * 1000
        
        data = response.json()
        
        return {
            'timestamp': datetime.now(),
            'success': data.get('success', False),
            'response_time_ms': elapsed
        }
    except Exception as e:
        return {
            'timestamp': datetime.now(),
            'success': False,
            'error': str(e)
        }

def continuous_monitoring(duration_minutes=10, check_interval=30):
    """Run monitoring for specified duration"""
    
    print("\n" + "="*60)
    print("👁️  API MONITORING STARTED")
    print("="*60)
    print(f"Duration: {duration_minutes} minutes")
    print(f"Check interval: {check_interval}s")
    print(f"Target: {API_URL}")
    print("\nPress Ctrl+C to stop\n")
    
    check_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_minutes * 60:
            check_count += 1
            
            health = health_check()
            prediction = prediction_test()
            
            status_symbol = "✅" if health['status'] == 'up' and prediction['success'] else "❌"
            
            print(f"{status_symbol} Check #{check_count} @ {health['timestamp'].strftime('%H:%M:%S')} - "
                  f"Status: {health['status']} | "
                  f"Response: {health.get('response_time_ms', 0):.0f}ms | "
                  f"Prediction: {'OK' if prediction['success'] else 'FAIL'}")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
    
    print(f"\nTotal checks: {check_count}")
    print("✅ Monitoring complete")

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("👁️  API MONITORING SYSTEM")
    print("="*60)
    
    print("\n[1] Single Health Check")
    print("[2] Continuous Monitoring (10 minutes)")
    print("[3] Custom Duration")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        health = health_check()
        prediction = prediction_test()
        
        print("\n🏥 HEALTH CHECK")
        print(f"Status: {health['status']}")
        print(f"Response time: {health.get('response_time_ms', 0):.0f}ms")
        print(f"Models loaded: {health.get('models_loaded', False)}")
        
        print("\n🔮 PREDICTION TEST")
        print(f"Success: {prediction['success']}")
        print(f"Response time: {prediction.get('response_time_ms', 0):.0f}ms")
        
    elif choice == '2':
        continuous_monitoring(duration_minutes=10, check_interval=30)
        
    elif choice == '3':
        duration = int(input("Duration (minutes): "))
        interval = int(input("Check interval (seconds): "))
        continuous_monitoring(duration_minutes=duration, check_interval=interval)
        
    else:
        print("Invalid choice.")