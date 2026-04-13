import torch
import torch.nn as nn
from policy_runner import PolicyRunner, PolicyMLP, RunningStandardScaler
import argparse
import os

class JITPolicy(nn.Module):
    def __init__(self, obs_dim, layers, action_dim):
        super().__init__()
        self.scaler = RunningStandardScaler(obs_dim, "cpu")
        self.policy = PolicyMLP(obs_dim, layers, action_dim)

    def forward(self, obs):
        # obs: [Batch, ObsDim]
        obs_norm = self.scaler(obs)
        return self.policy(obs_norm)

def export(checkpoint_path, out_path):
    print(f"[Export] Loading {checkpoint_path}...")
    
    # Use PolicyRunner's inspection logic
    dummy_runner = PolicyRunner(checkpoint_path, device="cpu")
    
    jit_model = JITPolicy(dummy_runner.obs_dim, dummy_runner.layers, dummy_runner.action_dim)
    jit_model.scaler.load_state_dict(dummy_runner.scaler.state_dict())
    jit_model.policy.load_state_dict(dummy_runner.policy.state_dict())
    jit_model.eval()

    print(f"[Export] Tracing model...")
    example_input = torch.randn(1, dummy_runner.obs_dim)
    traced_script_module = torch.jit.trace(jit_model, example_input)
    
    print(f"[Export] Saving to {out_path}...")
    traced_script_module.save(out_path)
    print("[Export] Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_agent.pt")
    parser.add_argument("--out", type=str, default="policy_jit.pt", help="Output path")
    args = parser.parse_args()
    
    export(args.checkpoint, args.out)
