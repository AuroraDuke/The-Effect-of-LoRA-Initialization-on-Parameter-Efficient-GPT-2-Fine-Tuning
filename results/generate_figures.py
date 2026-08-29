"""Generate task-specific figures from result CSV files.

Run from the project root:
  python results/generate_figures.py
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")


TASK_FILES = {
  "classifier": "classifier.csv",
  "paraphrase": "paraphrase_detection.csv",
  "sonnet": "sonnet_generation.csv",
}


def ensure_dirs():
  os.makedirs(FIGURES_DIR, exist_ok=True)
  os.makedirs(SUMMARY_DIR, exist_ok=True)


def read_csv(filename):
  path = os.path.join(RESULTS_DIR, filename)
  if not os.path.exists(path):
    print(f"Skipping missing file: {path}")
    return []

  with open(path, "r", encoding="utf-8", newline="") as f:
    return list(csv.DictReader(f))


def as_float(value):
  if value is None or value == "":
    return None
  try:
    return float(value)
  except ValueError:
    return None


def as_int(value):
  number = as_float(value)
  if number is None:
    return None
  return int(number)


def clean_label(*parts):
  values = [str(part) for part in parts if part not in (None, "")]
  return " | ".join(values) if values else "run"


def save_line_plot(rows, metric, title, ylabel, filename, group_fields):
  grouped = defaultdict(list)
  for row in rows:
    epoch = as_int(row.get("epoch"))
    value = as_float(row.get(metric))
    if epoch is None or value is None:
      continue

    label = clean_label(*[row.get(field) for field in group_fields])
    grouped[label].append((epoch, value))

  if not grouped:
    print(f"Skipping {filename}: no values for {metric}")
    return

  plt.figure(figsize=(10, 6))
  for label, points in sorted(grouped.items()):
    points = sorted(points)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.plot(xs, ys, marker="o", label=label)

  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel(ylabel)
  plt.grid(True, alpha=0.3)
  plt.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
  plt.close()
  print(f"Saved figure: {filename}")


def save_best_bar_plot(rows, metric, title, ylabel, filename, group_fields, higher_is_better=True):
  best_by_group = {}
  for row in rows:
    value = as_float(row.get(metric))
    if value is None:
      continue

    label = clean_label(*[row.get(field) for field in group_fields])
    if label not in best_by_group:
      best_by_group[label] = value
      continue

    current = best_by_group[label]
    if higher_is_better and value > current:
      best_by_group[label] = value
    elif not higher_is_better and value < current:
      best_by_group[label] = value

  if not best_by_group:
    print(f"Skipping {filename}: no values for {metric}")
    return

  labels = list(best_by_group.keys())
  values = [best_by_group[label] for label in labels]

  plt.figure(figsize=(max(8, len(labels) * 1.4), 6))
  plt.bar(labels, values)
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xticks(rotation=35, ha="right")
  plt.grid(True, axis="y", alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
  plt.close()
  print(f"Saved figure: {filename}")


def write_summary_csv(task_name, rows, group_fields, metric_fields):
  summaries = {}

  for row in rows:
    label = clean_label(*[row.get(field) for field in group_fields])
    if label not in summaries:
      summaries[label] = {field: row.get(field, "") for field in group_fields}

    for metric in metric_fields:
      value = as_float(row.get(metric))
      if value is None:
        continue
      best_key = f"best_{metric}"
      if best_key not in summaries[label] or value > summaries[label][best_key]:
        summaries[label][best_key] = value

  if not summaries:
    return

  fieldnames = list(group_fields) + [f"best_{metric}" for metric in metric_fields]
  path = os.path.join(SUMMARY_DIR, f"{task_name}_summary.csv")
  with open(path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summaries.values():
      writer.writerow({field: row.get(field, "") for field in fieldnames})

  print(f"Saved summary: {path}")


def generate_classifier_figures(rows):
  if not rows:
    return

  group_fields = ("data_label", "fine_tune_mode")
  save_line_plot(rows, "train_loss", "Classifier Train Loss by Epoch", "Train Loss", "classifier_train_loss.png", group_fields)
  save_line_plot(rows, "dev_acc", "Classifier Dev Accuracy by Epoch", "Dev Accuracy", "classifier_dev_accuracy.png", group_fields)
  save_line_plot(rows, "dev_f1", "Classifier Dev F1 by Epoch", "Dev F1", "classifier_dev_f1.png", group_fields)
  save_best_bar_plot(rows, "dev_acc", "Best Classifier Dev Accuracy", "Best Dev Accuracy", "classifier_best_dev_accuracy.png", group_fields)
  write_summary_csv("classifier", rows, group_fields, ("train_acc", "train_f1", "dev_acc", "dev_f1"))


def generate_paraphrase_figures(rows):
  if not rows:
    return

  group_fields = ("fine_tune_mode", "lora_init_method", "lora_r", "lora_alpha")
  save_line_plot(rows, "train_loss", "Paraphrase Train Loss by Epoch", "Train Loss", "paraphrase_train_loss.png", group_fields)
  save_line_plot(rows, "dev_acc", "Paraphrase Dev Accuracy by Epoch", "Dev Accuracy", "paraphrase_dev_accuracy.png", group_fields)
  save_line_plot(rows, "dev_f1", "Paraphrase Dev F1 by Epoch", "Dev F1", "paraphrase_dev_f1.png", group_fields)
  save_best_bar_plot(rows, "dev_acc", "Best Paraphrase Dev Accuracy", "Best Dev Accuracy", "paraphrase_best_dev_accuracy.png", group_fields)
  write_summary_csv("paraphrase", rows, group_fields, ("dev_acc", "dev_f1"))


def generate_sonnet_figures(rows):
  if not rows:
    return

  group_fields = ("fine_tune_mode", "lora_init_method", "lora_r", "lora_alpha")
  save_line_plot(rows, "train_loss", "Sonnet Train Loss by Epoch", "Train Loss", "sonnet_train_loss.png", group_fields)
  save_line_plot(rows, "chrF", "Sonnet chrF by Epoch", "chrF", "sonnet_chrf.png", group_fields)
  save_line_plot(rows, "BLEU", "Sonnet BLEU by Epoch", "BLEU", "sonnet_bleu.png", group_fields)
  save_line_plot(rows, "attention_std", "Attention Weight Std by Epoch", "Attention Std", "sonnet_attention_std.png", group_fields)
  save_line_plot(rows, "lora_a_std", "LoRA-A Weight Std by Epoch", "LoRA-A Std", "sonnet_lora_a_std.png", group_fields)
  save_line_plot(rows, "lora_b_std", "LoRA-B Weight Std by Epoch", "LoRA-B Std", "sonnet_lora_b_std.png", group_fields)
  save_best_bar_plot(rows, "chrF", "Best Sonnet chrF", "Best chrF", "sonnet_best_chrf.png", group_fields)
  save_best_bar_plot(rows, "BLEU", "Best Sonnet BLEU", "Best BLEU", "sonnet_best_bleu.png", group_fields)
  write_summary_csv("sonnet", rows, group_fields, ("chrF", "BLEU"))


def main():
  ensure_dirs()
  generate_classifier_figures(read_csv(TASK_FILES["classifier"]))
  generate_paraphrase_figures(read_csv(TASK_FILES["paraphrase"]))
  generate_sonnet_figures(read_csv(TASK_FILES["sonnet"]))


if __name__ == "__main__":
  main()
