import torch

from einops import rearrange
from torch import nn


def initialize_lora_default(lora_A: nn.Linear, lora_B: nn.Linear):
  nn.init.normal_(lora_A.weight, mean=0.0, std=0.02)
  nn.init.zeros_(lora_B.weight)


def initialize_lora_weight_dist(lora_A: nn.Linear, lora_B: nn.Linear, W: torch.Tensor, init_scale: float = 0.01):
  mean_W = W.mean().item()
  std_W = W.std().item()
  nn.init.normal_(lora_A.weight, mean=mean_W, std=std_W)
  nn.init.normal_(lora_B.weight, mean=mean_W, std=std_W)
  lora_A.weight.data.mul_(init_scale)
  lora_B.weight.data.mul_(init_scale)


def initialize_lora_svd(lora_A: nn.Linear, lora_B: nn.Linear, W: torch.Tensor, r: int, svd_scale: float = 0.01):
  if W.dim() != 2:
    raise ValueError(f"SVD initialization expects 2D weight matrix, got shape {tuple(W.shape)}")
  if r <= 0:
    raise ValueError("SVD initialization requires r > 0")

  U, S, Vh = torch.linalg.svd(W, full_matrices=False)
  if r > S.size(0):
    raise ValueError(f"SVD rank r={r} exceeds minimum dimension of W {S.size(0)}")

  U_r = U[:, :r]
  S_r = S[:r]
  V_r = Vh[:r, :]
  sqrt_s = torch.sqrt(S_r)

  lora_B.weight.data.copy_((U_r * sqrt_s.unsqueeze(0)) * svd_scale)
  lora_A.weight.data.copy_((sqrt_s.unsqueeze(1) * V_r) * svd_scale)


def _maybe_transpose_weight(W: torch.Tensor, lora_A: nn.Linear, lora_B: nn.Linear, layer_name: str):
  if W.dim() != 2:
    print(f"[LoRA Init] layer={layer_name} warning: expected 2D W but got shape {tuple(W.shape)}")
    return W

  out_features = lora_B.weight.shape[0]
  rank = lora_B.weight.shape[1]
  in_features = lora_A.weight.shape[1]

  if W.shape == (out_features, in_features):
    return W
  if W.shape == (in_features, out_features):
    print(f"[LoRA Init] layer={layer_name} W shape {tuple(W.shape)} is transposed relative to expected {(out_features, in_features)}; using W.T")
    return W.T

  print(
    f"[LoRA Init] layer={layer_name} warning: W shape {tuple(W.shape)} did not match expected "
    f"{(out_features, in_features)}; using W as-is"
  )
  return W


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    self.lora_r = getattr(config, "lora_r", 0)
    self.lora_alpha = getattr(config, "lora_alpha", 1.0)
    self.lora_init = getattr(config, "lora_init", "normal")
    self.lora_init_method = getattr(config, "lora_init_method", "default")
    self.lora_init_scale = getattr(config, "lora_init_scale", 0.01)
    self.lora_svd_scale = getattr(config, "lora_svd_scale", 0.01)
    self.lora_scaling = self.lora_alpha / self.lora_r if self.lora_r > 0 else 1.0

    self.q_lora_a = None
    self.q_lora_b = None
    self.k_lora_a = None
    self.k_lora_b = None
    self.v_lora_a = None
    self.v_lora_b = None

    if self.lora_r > 0:
      self.q_lora_a = nn.Linear(config.hidden_size, self.lora_r, bias=False)
      self.q_lora_b = nn.Linear(self.lora_r, self.all_head_size, bias=False)
      self.k_lora_a = nn.Linear(config.hidden_size, self.lora_r, bias=False)
      self.k_lora_b = nn.Linear(self.lora_r, self.all_head_size, bias=False)
      self.v_lora_a = nn.Linear(config.hidden_size, self.lora_r, bias=False)
      self.v_lora_b = nn.Linear(self.lora_r, self.all_head_size, bias=False)
      if self.lora_init_method == "default":
        self._init_lora_weights()
      else:
        for module in [self.q_lora_a, self.k_lora_a, self.v_lora_a, self.q_lora_b, self.k_lora_b, self.v_lora_b]:
          nn.init.zeros_(module.weight)

  def _init_lora_weights(self):
    def init_a(tensor):
      if self.lora_init == "normal":
        return nn.init.normal_(tensor, mean=0.0, std=0.02)
      if self.lora_init == "uniform":
        return nn.init.uniform_(tensor, a=-0.02, b=0.02)
      if self.lora_init == "zeros":
        return nn.init.zeros_(tensor)
      if self.lora_init == "xavier_uniform":
        return nn.init.xavier_uniform_(tensor)
      raise ValueError(f"Unsupported LoRA init: {self.lora_init}")

    for module in [self.q_lora_a, self.k_lora_a, self.v_lora_a]:
      init_a(module.weight)
    for module in [self.q_lora_b, self.k_lora_b, self.v_lora_b]:
      nn.init.zeros_(module.weight)

  def initialize_lora_from_weight(self, W: torch.Tensor, pair_name: str):
    if self.lora_r <= 0:
      return

    lora_A = getattr(self, f"{pair_name}_lora_a")
    lora_B = getattr(self, f"{pair_name}_lora_b")
    if lora_A is None or lora_B is None:
      return

    layer_name = getattr(self, "layer_name", "unknown")
    W = _maybe_transpose_weight(W, lora_A, lora_B, f"{layer_name}.{pair_name}")

    if self.lora_init_method == "default":
      initialize_lora_default(lora_A, lora_B)
    elif self.lora_init_method == "weight_dist":
      initialize_lora_weight_dist(lora_A, lora_B, W, init_scale=self.lora_init_scale)
    elif self.lora_init_method == "svd":
      initialize_lora_svd(lora_A, lora_B, W, self.lora_r, svd_scale=self.lora_svd_scale)
    else:
      raise ValueError(f"Unsupported LoRA init method: {self.lora_init_method}")

  def transform(self, x, linear_layer, lora_a=None, lora_b=None):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    if lora_a is not None and lora_b is not None and self.lora_r > 0:
      proj = proj + self.lora_scaling * lora_b(lora_a(x))
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key, query, value: [batch_size, num_heads, seq_len, head_size]
    attention_mask: [batch_size, 1, 1, seq_len]
    """
    #ENTER YOUR CODE
    
    # Scaled dot-product attention scores.
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / (self.attention_head_size ** 0.5)

    # Causal mask to prevent attending to future positions.
    seq_len = scores.size(-1)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    scores = scores.masked_fill(~causal_mask, float('-10000.0'))

    # Add the provided attention mask (padding mask) and normalize.
    scores = scores + attention_mask
    attention_probs = torch.softmax(scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    # Weighted sum of values.
    context = torch.matmul(attention_probs, value)
    context = rearrange(context, 'b h t d -> b t (h d)')
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key, self.k_lora_a, self.k_lora_b)
    value_layer = self.transform(hidden_states, self.value, self.v_lora_a, self.v_lora_b)
    query_layer = self.transform(hidden_states, self.query, self.q_lora_a, self.q_lora_b)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
