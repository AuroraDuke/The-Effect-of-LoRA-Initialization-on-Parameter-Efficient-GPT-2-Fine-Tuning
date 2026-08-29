# Run Commands

Run these commands from the project root:

```powershell
cd C:\Users\Victus\Desktop\ceng534
```

## Sonnet Generation

Full fine-tuning on GPU:

```powershell
python sonnet_generation.py --use_gpu --fine_tune_mode full --epochs 10 --batch_size 8 --lr 1e-5
```

Last-layer fine-tuning on GPU:

```powershell
python sonnet_generation.py --use_gpu --fine_tune_mode last-layer --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA fine-tuning on GPU:

```powershell
python sonnet_generation.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method default --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA with weight-distribution initialization:

```powershell
python sonnet_generation.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method weight_dist --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA with SVD initialization:

```powershell
python sonnet_generation.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method svd --epochs 10 --batch_size 8 --lr 1e-5
```

CPU quick run:

```powershell
python sonnet_generation.py --fine_tune_mode lora --lora_r 4 --lora_alpha 32 --epochs 1 --batch_size 2 --lr 1e-5
```

Sonnet checkpoints are saved under:

```text
results/sonnet_generation_checkpoints/
```

## Paraphrase Detection

Full fine-tuning on GPU:

```powershell
python paraphrase_detection.py --use_gpu --fine_tune_mode full --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA fine-tuning on GPU:

```powershell
python paraphrase_detection.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method default --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA with weight-distribution initialization:

```powershell
python paraphrase_detection.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method weight_dist --epochs 10 --batch_size 8 --lr 1e-5
```

LoRA with SVD initialization:

```powershell
python paraphrase_detection.py --use_gpu --fine_tune_mode lora --lora_r 8 --lora_alpha 16 --lora_init_method svd --epochs 10 --batch_size 8 --lr 1e-5
```

CPU quick run:

```powershell
python paraphrase_detection.py --fine_tune_mode lora --lora_r 4 --lora_alpha 32 --epochs 1 --batch_size 2 --lr 1e-5
```

## Sentiment Classification

Last-linear-layer fine-tuning on GPU:

```powershell
python classifier.py --use_gpu --fine-tune-mode last-linear-layer --epochs 10 --batch_size 8 --lr 1e-3
```

Full-model fine-tuning on GPU:

```powershell
python classifier.py --use_gpu --fine-tune-mode full-model --epochs 10 --batch_size 8 --lr 1e-5
```

CPU quick run:

```powershell
python classifier.py --fine-tune-mode last-linear-layer --epochs 1 --batch_size 2 --lr 1e-3
```

## Useful Checks

Optimizer test:

```powershell
python scripts/optimizer_test.py
```

GPT-2 sanity check:

```powershell
python scripts/sanity_check.py
```

LoRA initialization demo:

```powershell
python scripts/test_lora_init.py
```

Prepare submission zip:

```powershell
python scripts/prepare_submit.py
```

Generate result figures:

```powershell
python results/generate_figures.py
```
