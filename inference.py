import requests
import sys

def main():
    base_url = "http://127.0.0.1:7860"
    
    print("[START]")
    try:
        # 1. Reset
        res_reset = requests.post(f"{base_url}/reset", json={"task_id": 1, "seed": 42})
        res_reset.raise_for_status()
        
        print("[STEP]")
        # 2. Step
        res_step = requests.post(f"{base_url}/step", json={"action_type": "query_counterfactual", "service_id": "api-gateway"})
        res_step.raise_for_status()
        
        # 3. State
        res_state = requests.get(f"{base_url}/state")
        res_state.raise_for_status()
        
        print("[END]")
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
