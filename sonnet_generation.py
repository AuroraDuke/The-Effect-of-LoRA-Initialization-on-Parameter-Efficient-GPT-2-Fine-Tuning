'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import os
import random
import torch
import time

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from src.datasets import (
  SonnetsDataset,
)
from src.models.gpt2 import GPT2Model

from src.optimizer import AdamW
from src.evaluation import test_sonnet
from src.lora_attention import print_attention_and_lora_stats, get_attention_and_lora_stats
from src.log_experiments import log_sonnet_epoch

TQDM_DISABLE = False


def make_safe_name(value):
  return str(value).replace('/', '-').replace('\\', '-').replace('.', 'p')


def build_checkpoint_dir(args):
  run_parts = [
    'sonnet_generation',
    args.fine_tune_mode,
    args.model_size,
    f'epochs{args.epochs}',
    f'lr{make_safe_name(args.lr)}',
  ]
  if args.fine_tune_mode == 'lora':
    run_parts.extend([
      f'r{args.lora_r}',
      f'alpha{make_safe_name(args.lora_alpha)}',
      args.lora_init_method,
    ])
  return os.path.join('results', 'sonnet_generation_checkpoints', '_'.join(run_parts))


def get_checkpoint_path(args, epoch):
  return os.path.join(args.checkpoint_dir, f'epoch_{epoch}_{args.filepath}')


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size,
      d=args.d,
      l=args.l,
      num_heads=args.num_heads,
      lora_r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_init=args.lora_init,
      lora_init_method=args.lora_init_method,
      lora_init_scale=args.lora_init_scale,
      lora_svd_scale=args.lora_svd_scale,
    )
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    if args.fine_tune_mode == 'full':
      for param in self.gpt.parameters():
        param.requires_grad = True
    elif args.fine_tune_mode == 'last-layer':
      for param in self.gpt.parameters():
        param.requires_grad = False
      for param in self.gpt.gpt_layers[-1].parameters():
        param.requires_grad = True
      for param in self.gpt.final_layer_norm.parameters():
        param.requires_grad = True
    elif args.fine_tune_mode == 'lora':
      if args.lora_r <= 0:
        raise ValueError('LoRA mode requires --lora_r > 0')
      for param in self.gpt.parameters():
        param.requires_grad = False
      for name, param in self.gpt.named_parameters():
        if "lora_" in name:
          param.requires_grad = True
    else:
      raise ValueError(f'Unsupported fine-tune mode: {args.fine_tune_mode}')

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    #YOUR CODE HERE
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    epoch_start = time.time()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      generated_output = output[1]
      generated_sonnets.append((batch[0], generated_output))

    with open(args.sonnet_out, "w+", encoding="utf-8") as f:
      f.write(f"--Generated Sonnets-- \n\n")
      for sonnet in generated_sonnets:
        f.write(f"\n{sonnet[0]}\n")
        f.write(sonnet[1])

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    chrF = None
    BLEU = None
    note = None
    try:
      scores = test_sonnet(test_path=args.sonnet_out, gold_path=args.true_sonnet_path)
      chrF = scores['chrf']
      BLEU = scores['bleu']
      print(f"Epoch {epoch}: chrF :: {chrF:.2f}, BLEU :: {BLEU:.2f}")
    except Exception as e:
      note = f"Evaluation skipped: {e}"
      print(f"Skipping epoch chrF/BLEU evaluation: {e}")

    lora_stats = get_attention_and_lora_stats(model) if args.fine_tune_mode == 'lora' else None
    log_sonnet_epoch(args, epoch, train_loss, chrF, BLEU, lora_stats=lora_stats, training_time=epoch_time, note=note)
    print('Saving the model...')
    save_model(model, optimizer, args, get_checkpoint_path(args, epoch))
    print_attention_and_lora_stats(model, args)


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(get_checkpoint_path(args, args.epochs - 1), weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  print_attention_and_lora_stats(model, saved['args'])



def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank; use 0 to disable LoRA.")
   # if you set this to 0, the model will be trained in full fine-tuning mode.
   # and you not see any LoRA statistics in the logs if you do this.
  parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA scaling alpha.")
  parser.add_argument("--lora_init", type=str, choices=['normal','uniform','zeros','xavier_uniform'],
                      default='normal', help="Initialization for LoRA A matrices.")
  parser.add_argument("--lora_init_method", type=str, choices=['default','weight_dist','svd'],
                      default='default', help="Method used to initialize LoRA matrices.")
  parser.add_argument("--lora_init_scale", type=float, default=0.01,
                      help="Scale factor for weight distribution LoRA initialization.")
  parser.add_argument("--lora_svd_scale", type=float, default=0.01,
                      help="Scale factor for SVD-based LoRA initialization.")

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--fine_tune_mode", type=str, choices=['full', 'last-layer', 'lora'], default='full',
                      help="Whether to fine-tune the full model, only the last transformer layer, or only LoRA adapters.")
  parser.add_argument("--true_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt",
                      help="Gold sonnets file used to compute chrF for generated output.")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # If we are not in LoRA fine-tune mode, force lora_r to 0 so adapters are disabled.
  if args.fine_tune_mode in ('full', 'last-layer'):
    args.lora_r = 0

  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  args.checkpoint_dir = build_checkpoint_dir(args)
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)
  try:
    scores = test_sonnet(test_path=args.sonnet_out, gold_path=args.true_sonnet_path)
    print(f"Final chrF: {scores['chrf']:.2f}, BLEU: {scores['bleu']:.2f}")
  except Exception as e:
    print(f"Could not compute chrF/BLEU via evaluation.test_sonnet: {e}")

 
