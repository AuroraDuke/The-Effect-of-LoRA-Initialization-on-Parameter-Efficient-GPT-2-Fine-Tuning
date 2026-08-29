"""Distribution estimation helpers for model weight tensors."""

import torch


def estimate_distribution(tensor: torch.Tensor):
  """Estimate the closest simple distribution family for a tensor.

  The estimate is heuristic: it uses skewness, excess kurtosis, and the
  range/std ratio to label weights as zero-like, normal-like, uniform-like, or
  other.
  """
  values = tensor.detach().float().flatten()
  if values.numel() == 0:
    return {
      'name': 'empty',
      'skewness': 0.0,
      'excess_kurtosis': 0.0,
    }

  mean = values.mean()
  std = values.std(unbiased=False)
  min_value = values.min()
  max_value = values.max()

  if std.item() < 1e-12:
    return {
      'name': 'zero-like' if values.norm().item() < 1e-12 else 'constant-like',
      'skewness': 0.0,
      'excess_kurtosis': 0.0,
    }

  centered = values - mean
  normalized = centered / std
  skewness = torch.mean(normalized ** 3).item()
  excess_kurtosis = torch.mean(normalized ** 4).item() - 3.0
  range_to_std = ((max_value - min_value) / std).item()

  if abs(skewness) < 0.35 and abs(excess_kurtosis) < 0.75:
    name = 'normal-like'
  elif abs(skewness) < 0.35 and abs(excess_kurtosis + 1.2) < 0.45 and 2.8 <= range_to_std <= 4.2:
    name = 'uniform-like'
  else:
    name = 'other'

  return {
    'name': name,
    'skewness': skewness,
    'excess_kurtosis': excess_kurtosis,
  }
