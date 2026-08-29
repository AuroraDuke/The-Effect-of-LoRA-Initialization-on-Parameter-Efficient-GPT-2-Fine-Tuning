"""LoRA-related utilities for attention and weight statistics."""

import torch

from src.distribution_metrics import estimate_distribution


def get_attention_and_lora_stats(model):
  """Return aggregated attention and LoRA weight statistics."""
  attention_weights = []
  lora_a_weights = []
  lora_b_weights = []

  for name, param in model.gpt.named_parameters():
    if param.dim() < 1 or 'bias' in name:
      continue

    if 'self_attention' in name and ('query' in name or 'key' in name or 'value' in name):
      if 'lora' not in name:
        attention_weights.append(param.data)
    elif 'lora_a' in name:
      lora_a_weights.append(param.data)
    elif 'lora_b' in name:
      lora_b_weights.append(param.data)

  stats = {}
  if attention_weights:
    attn_concat = torch.cat([w.flatten() for w in attention_weights])
    attn_dist = estimate_distribution(attn_concat)
    stats.update({
      'attention_mean': attn_concat.mean().item(),
      'attention_std': attn_concat.std().item(),
      'attention_min': attn_concat.min().item(),
      'attention_max': attn_concat.max().item(),
      'attention_norm': attn_concat.norm().item(),
      'attention_distribution': attn_dist['name'],
      'attention_skewness': attn_dist['skewness'],
      'attention_excess_kurtosis': attn_dist['excess_kurtosis'],
    })
  if lora_a_weights:
    lora_a_concat = torch.cat([w.flatten() for w in lora_a_weights])
    lora_a_dist = estimate_distribution(lora_a_concat)
    stats.update({
      'lora_a_mean': lora_a_concat.mean().item(),
      'lora_a_std': lora_a_concat.std().item(),
      'lora_a_min': lora_a_concat.min().item(),
      'lora_a_max': lora_a_concat.max().item(),
      'lora_a_norm': lora_a_concat.norm().item(),
      'lora_a_distribution': lora_a_dist['name'],
      'lora_a_skewness': lora_a_dist['skewness'],
      'lora_a_excess_kurtosis': lora_a_dist['excess_kurtosis'],
    })
  if lora_b_weights:
    lora_b_concat = torch.cat([w.flatten() for w in lora_b_weights])
    lora_b_dist = estimate_distribution(lora_b_concat)
    stats.update({
      'lora_b_mean': lora_b_concat.mean().item(),
      'lora_b_std': lora_b_concat.std().item(),
      'lora_b_min': lora_b_concat.min().item(),
      'lora_b_max': lora_b_concat.max().item(),
      'lora_b_norm': lora_b_concat.norm().item(),
      'lora_b_distribution': lora_b_dist['name'],
      'lora_b_skewness': lora_b_dist['skewness'],
      'lora_b_excess_kurtosis': lora_b_dist['excess_kurtosis'],
    })
  return stats


def print_attention_and_lora_stats(model, args):
  """Print aggregated statistics of attention and LoRA weights.
  
  Prints mean, std, min, max, and norm for:
  - All attention layer weights (Query, Key, Value combined)
  - All LoRA A matrices combined
  - All LoRA B matrices combined
  
  Args:
    model: SonnetGPT model instance
    args: Arguments object containing fine_tune_mode flag
  """
  if args.fine_tune_mode != 'lora':
    return

  stats = get_attention_and_lora_stats(model)
  if not stats:
    return

  print("\n" + "="*110)
  print("Attention & LoRA Weight Statistics (Aggregated)")
  print("="*110)
  print(
    f"[Attention] mean={stats['attention_mean']:.6f} std={stats['attention_std']:.6f} "
    f"min={stats['attention_min']:.6f} max={stats['attention_max']:.6f} "
    f"norm={stats['attention_norm']:.4f} dist={stats['attention_distribution']}"
  )
  print(
    f"[LoRA-A]    mean={stats['lora_a_mean']:.6f} std={stats['lora_a_std']:.6f} "
    f"min={stats['lora_a_min']:.6f} max={stats['lora_a_max']:.6f} "
    f"norm={stats['lora_a_norm']:.4f} dist={stats['lora_a_distribution']}"
  )
  print(
    f"[LoRA-B]    mean={stats['lora_b_mean']:.6f} std={stats['lora_b_std']:.6f} "
    f"min={stats['lora_b_min']:.6f} max={stats['lora_b_max']:.6f} "
    f"norm={stats['lora_b_norm']:.4f} dist={stats['lora_b_distribution']}"
  )
  print("="*110 + "\n")
