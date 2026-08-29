"""Experiment logging utility for sonnet, paraphrase, and classifier workflows."""

import csv
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
CSV_FILES = {
  'sonnet': 'sonnet_generation.csv',
  'paraphrase': 'paraphrase_detection.csv',
  'classifier': 'classifier.csv',
}

SONNET_FIELDS = [
  'timestamp', 'task', 'run_name', 'epoch', 'train_loss', 'chrF', 'BLEU',
  'model_size', 'lora_r', 'lora_alpha', 'lora_init_method', 'lora_init_scale', 'lora_svd_scale',
  'batch_size', 'lr', 'epochs', 'temperature', 'top_p', 'fine_tune_mode',
  'attention_mean', 'attention_std', 'lora_a_mean', 'lora_a_std', 'lora_b_mean', 'lora_b_std',
  'training_time',
  'note'
]

PARAPHRASE_FIELDS = [
  'timestamp', 'task', 'run_name', 'epoch', 'train_loss', 'dev_acc', 'dev_f1',
  'model_size', 'batch_size', 'lr', 'epochs', 'fine_tune_mode',
  'lora_r', 'lora_alpha', 'lora_init_method', 'lora_init_scale', 'lora_svd_scale',
  'attention_mean', 'attention_std', 'lora_a_mean', 'lora_a_std', 'lora_b_mean', 'lora_b_std',
  'training_time',
  'para_train', 'para_dev', 'para_test', 'note'
]

CLASSIFIER_FIELDS = [
  'timestamp', 'task', 'run_name', 'epoch', 'train_loss', 'train_acc', 'train_f1', 'dev_acc', 'dev_f1',
  'fine_tune_mode', 'lr', 'batch_size', 'epochs', 'hidden_dropout_prob',
  'data_label', 'train_data', 'dev_data', 'test_data', 'note'
]


def _ensure_results_dir():
  os.makedirs(RESULTS_DIR, exist_ok=True)


def _to_dict(args, keys):
  result = {}
  for key in keys:
    value = None
    if isinstance(args, dict):
      value = args.get(key, None)
    else:
      value = getattr(args, key, None)
    if isinstance(value, (dict, list)):
      value = json.dumps(value, ensure_ascii=False)
    result[key] = value
  return result


def _clean_row(row, fieldnames):
  return {k: '' if row.get(k) is None else row.get(k) for k in fieldnames}


def _append_row(filename, fieldnames, row):
  _ensure_results_dir()
  path = os.path.join(RESULTS_DIR, filename)
  write_header = not os.path.exists(path)
  with open(path, 'a', encoding='utf-8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
    writer.writerow(_clean_row(row, fieldnames))


def log_sonnet_epoch(args, epoch, train_loss, chrF, BLEU, lora_stats=None, training_time=None, note=None):
  row = {
    'timestamp': datetime.utcnow().isoformat(),
    'task': 'sonnet_generation',
    'run_name': getattr(args, 'filepath', None) or getattr(args, 'sonnet_out', None) or 'sonnet_run',
    'epoch': epoch,
    'train_loss': train_loss,
    'chrF': chrF,
    'BLEU': BLEU,
    'note': note,
    'training_time': training_time,
  }
  row.update(_to_dict(args, [
    'model_size', 'lora_r', 'lora_alpha', 'lora_init_method', 'lora_init_scale', 'lora_svd_scale',
    'batch_size', 'lr', 'epochs', 'temperature', 'top_p', 'fine_tune_mode'
  ]))
  if lora_stats is not None:
    row.update({
      'attention_mean': lora_stats.get('attention_mean'),
      'attention_std': lora_stats.get('attention_std'),
      'lora_a_mean': lora_stats.get('lora_a_mean'),
      'lora_a_std': lora_stats.get('lora_a_std'),
      'lora_b_mean': lora_stats.get('lora_b_mean'),
      'lora_b_std': lora_stats.get('lora_b_std'),
    })
  _append_row(CSV_FILES['sonnet'], SONNET_FIELDS, row)


def log_paraphrase_epoch(args, epoch, train_loss, dev_acc, dev_f1, lora_stats=None, training_time=None, note=None):
  row = {
    'timestamp': datetime.utcnow().isoformat(),
    'task': 'paraphrase_detection',
    'run_name': getattr(args, 'filepath', None) or 'paraphrase_run',
    'epoch': epoch,
    'train_loss': train_loss,
    'dev_acc': dev_acc,
    'dev_f1': dev_f1,
    'note': note,
    'training_time': training_time,
  }
  row.update(_to_dict(args, [
    'model_size', 'batch_size', 'lr', 'epochs', 'fine_tune_mode',
    'lora_r', 'lora_alpha', 'lora_init_method', 'lora_init_scale', 'lora_svd_scale',
    'para_train', 'para_dev', 'para_test'
  ]))
  if lora_stats is not None:
    row.update({
      'attention_mean': lora_stats.get('attention_mean'),
      'attention_std': lora_stats.get('attention_std'),
      'lora_a_mean': lora_stats.get('lora_a_mean'),
      'lora_a_std': lora_stats.get('lora_a_std'),
      'lora_b_mean': lora_stats.get('lora_b_mean'),
      'lora_b_std': lora_stats.get('lora_b_std'),
    })
  _append_row(CSV_FILES['paraphrase'], PARAPHRASE_FIELDS, row)


def log_classifier_epoch(config, epoch, train_loss, train_acc, train_f1, dev_acc, dev_f1, data_label, note=None):
  row = {
    'timestamp': datetime.utcnow().isoformat(),
    'task': 'classifier',
    'run_name': getattr(config, 'filepath', None) or f'classifier_{data_label}',
    'epoch': epoch,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'train_f1': train_f1,
    'dev_acc': dev_acc,
    'dev_f1': dev_f1,
    'data_label': data_label,
    'train_data': getattr(config, 'train', None),
    'dev_data': getattr(config, 'dev', None),
    'test_data': getattr(config, 'test', None),
    'note': note,
  }
  row.update(_to_dict(config, [
    'fine_tune_mode', 'lr', 'batch_size', 'epochs', 'hidden_dropout_prob'
  ]))
  _append_row(CSV_FILES['classifier'], CLASSIFIER_FIELDS, row)
