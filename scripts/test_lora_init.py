"""
Test script to demonstrate LoRA initialization methods with model-like distribution.
Run: python scripts/test_lora_init.py
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

def analyze_weight_distribution(weights, name):
  """Analyze and print weight distribution statistics."""
  print(f"\n{name}:")
  print(f"  Shape: {weights.shape}")
  print(f"  Mean:  {weights.mean().item():.6f}")
  print(f"  Std:   {weights.std().item():.6f}")
  print(f"  Min:   {weights.min().item():.6f}")
  print(f"  Max:   {weights.max().item():.6f}")
  return weights.mean().item(), weights.std().item()


def test_lora_inits():
  """Test different LoRA initialization methods."""
  print("="*80)
  print("LoRA Initialization Methods Comparison")
  print("="*80)
  
  # Load pretrained GPT-2
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"\nLoading GPT-2 on {device}...")
  model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
  
  # Get a sample Q,K,V weight from attention layer
  W_sample = model.transformer.h[0].attn.c_attn.weight.data
  analyze_weight_distribution(W_sample, "Pretrained c_attn weight (original)")
  
  # LoRA parameters
  lora_r = 8
  alpha = 16.0
  scale = alpha / lora_r
  
  print(f"\nLoRA Parameters: rank={lora_r}, alpha={alpha}, scale={scale:.4f}")
  print("="*80)
  
  # 1. Default initialization (Gaussian)
  print("\n1. DEFAULT INITIALIZATION (Gaussian)")
  print("-" * 80)
  lora_A_default = torch.zeros(lora_r, W_sample.shape[1], device=device)
  lora_B_default = torch.zeros(W_sample.shape[0], lora_r, device=device)
  
  torch.nn.init.normal_(lora_A_default, mean=0, std=0.02)
  torch.nn.init.zeros_(lora_B_default)
  
  analyze_weight_distribution(lora_A_default, "LoRA A (default - normal 0.02)")
  analyze_weight_distribution(lora_B_default, "LoRA B (default - zeros)")
  
  output_default = W_sample + scale * (lora_B_default @ lora_A_default)
  analyze_weight_distribution(output_default, "Output W + LoRA (default)")
  
  # 2. Weight Distribution initialization
  print("\n2. WEIGHT_DIST INITIALIZATION (matching pretrained distribution)")
  print("-" * 80)
  W_mean, W_std = analyze_weight_distribution(W_sample, "Using distribution from W")
  
  init_scale = 0.01
  lora_A_dist = torch.randn_like(lora_A_default) * (W_std * init_scale) + (W_mean * init_scale)
  lora_B_dist = torch.randn_like(lora_B_default) * (W_std * init_scale) + (W_mean * init_scale)
  
  analyze_weight_distribution(lora_A_dist, f"LoRA A (weight_dist - mean scaled by {init_scale})")
  analyze_weight_distribution(lora_B_dist, f"LoRA B (weight_dist - mean scaled by {init_scale})")
  
  output_dist = W_sample + scale * (lora_B_dist @ lora_A_dist)
  analyze_weight_distribution(output_dist, "Output W + LoRA (weight_dist)")
  
  # 3. SVD initialization
  print("\n3. SVD INITIALIZATION")
  print("-" * 80)
  U, S, Vt = torch.linalg.svd(W_sample, full_matrices=False)
  
  svd_scale = 0.01
  S_r = S[:lora_r]
  U_r = U[:, :lora_r]
  V_r = Vt[:lora_r, :]
  
  lora_B_svd = U_r @ torch.diag(torch.sqrt(S_r)) * svd_scale
  lora_A_svd = torch.diag(torch.sqrt(S_r)) @ V_r * svd_scale
  
  analyze_weight_distribution(lora_B_svd, f"LoRA B (SVD - scaled by {svd_scale})")
  analyze_weight_distribution(lora_A_svd, f"LoRA A (SVD - scaled by {svd_scale})")
  
  output_svd = W_sample + scale * (lora_B_svd @ lora_A_svd)
  analyze_weight_distribution(output_svd, "Output W + LoRA (SVD)")
  
  # Summary
  print("\n" + "="*80)
  print("SUMMARY - How close are outputs to original W?")
  print("="*80)
  
  diff_default = (output_default - W_sample).norm().item()
  diff_dist = (output_dist - W_sample).norm().item()
  diff_svd = (output_svd - W_sample).norm().item()
  
  print(f"Default diff from original:      {diff_default:.6f}")
  print(f"Weight_dist diff from original:  {diff_dist:.6f}")
  print(f"SVD diff from original:          {diff_svd:.6f}")
  print(f"\nBest: {'SVD' if diff_svd < min(diff_default, diff_dist) else 'Weight_dist' if diff_dist < diff_default else 'Default'}")
  
  print("\n✓ Test completed successfully!")


if __name__ == "__main__":
  test_lora_inits()
