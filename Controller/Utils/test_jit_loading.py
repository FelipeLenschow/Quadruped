import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Controller.policy_runner import PolicyRunner

def test_jit():
    checkpoint = "Controller/policy_walk_jit.pt"
    if not os.path.exists(checkpoint):
        print(f"Error: {checkpoint} not found")
        return

    # Set obs dim for Walk
    os.environ["QUADRUPED_OBS_DIM"] = "49"
    runner = PolicyRunner(checkpoint)
    
    # Dummy observation
    obs = np.zeros(49, dtype=np.float32)
    action = runner.get_action(obs)
    
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
    if action.shape == (12,):
        print("JIT Loading Test Passed!")
    else:
        print("JIT Loading Test Failed!")

if __name__ == "__main__":
    test_jit()
