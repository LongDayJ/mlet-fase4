"""Fine-tuning LoRA do GPT-2 (roda bem na T4 gratuita do Colab, ~10-20 min).

    python train.py --lang pt             # base pierreguillou/gpt2-small-portuguese -> adapter/pt
    python train.py --lang en             # base gpt2 -> adapter/en
    python train.py --lang pt --epochs 5 --rank 64 --lr 3e-4

Cada idioma tem a sua base e o seu adaptador (~20 MB) e treina só com os exemplos do idioma.
Gera adapter/<lang>/ e results/train_metrics_<lang>.json.
"""
import argparse
import json
import math
import os
import shutil

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from common import BASES, adapter_path, encode_example, make_collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(BASES), help="idioma do adaptador: pt ou en")
    ap.add_argument("--base", default=None, help="padrão: a base do idioma em common.BASES")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default=None, help="padrão: adapter/<lang>")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=None, help="padrão: min(768, contexto do modelo)")
    ap.add_argument("--limit", type=int, default=None, help="usar só N exemplos de treino (teste rápido)")
    ap.add_argument("--save-steps", type=int, default=25, help="salva um ponto de retomada a cada N passos")
    ap.add_argument("--ckpt-dir", default=None, help="padrão: outputs/<lang>")
    ap.add_argument("--force", action="store_true", help="retreina mesmo que o adaptador já exista (ignora pontos de retomada)")
    a = ap.parse_args()
    a.base = a.base or BASES[a.lang]
    a.out = a.out or adapter_path(a.lang)
    a.ckpt_dir = a.ckpt_dir or os.path.join("outputs", a.lang)
    print(f"idioma={a.lang} | base={a.base} | saída={a.out}")

    metrics_file = os.path.join(a.results, f"train_metrics_{a.lang}.json")
    if a.force:
        shutil.rmtree(a.ckpt_dir, ignore_errors=True)
    elif os.path.exists(os.path.join(a.out, "adapter_model.safetensors")) and os.path.exists(metrics_file):
        print(f"adaptador '{a.lang}' já treinado em {a.out}/ — nada a fazer (use --force para retreinar).")
        return

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base)
    n_pos = getattr(model.config, "n_positions", 1024)
    a.max_len = min(a.max_len or 768, n_pos)  # o modo família tem textos mais longos que os outros
    print(f"eos_token_id={tok.eos_token_id} | contexto do modelo={n_pos} | max_len de treino={a.max_len}")

    lora = LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"], fan_in_fan_out=True,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": a.train, "validation": a.val})
    ds = ds.filter(lambda ex: ex["lang"] == a.lang)  # cada adaptador só vê o seu idioma
    if a.limit:
        ds["train"] = ds["train"].select(range(min(a.limit, len(ds["train"]))))
    print(f"exemplos: treino={len(ds['train'])} validação={len(ds['validation'])}")
    ds = ds.map(
        lambda ex: encode_example(tok, ex["prompt"], ex["completion"], a.max_len),
        remove_columns=ds["train"].column_names,
    )

    total_steps = math.ceil(len(ds["train"]) / a.batch) * a.epochs
    warmup = max(1, int(0.05 * total_steps))  # inteiro: funciona em qualquer versão do transformers

    cut = sum(1 for ids in ds["train"]["labels"] if ids[-1] != tok.eos_token_id)
    print(f"exemplos cortados (perderam o EOS): {cut}/{len(ds['train'])} ({100 * cut / len(ds['train']):.1f}%)")
    if cut > 0.05 * len(ds["train"]):
        print("AVISO: muitos exemplos cortados; o modelo pode não aprender a parar. Reduza o tamanho dos textos do dataset.")

    args = TrainingArguments(
        output_dir=a.ckpt_dir, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        learning_rate=a.lr, num_train_epochs=a.epochs, warmup_steps=warmup, lr_scheduler_type="cosine",
        weight_decay=0.01, fp16=torch.cuda.is_available(), eval_strategy="epoch",
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=1,  # pontos de retomada
        logging_steps=20, report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=make_collate(tok),
    )
    last = get_last_checkpoint(a.ckpt_dir) if os.path.isdir(a.ckpt_dir) else None
    if last:
        print(f"RETOMANDO de {last} (mantenha os mesmos --epochs/--batch/--lr da execução anterior)")
    trainer.train(resume_from_checkpoint=last)
    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
    print(metrics)

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    os.makedirs(a.results, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump({"base": a.base, "args": vars(a), "metrics": metrics, "log": trainer.state.log_history}, f, indent=2)
    shutil.rmtree(a.ckpt_dir, ignore_errors=True)  # treino concluído: pontos de retomada não servem mais
    print(f"adaptador salvo em {a.out}/")


if __name__ == "__main__":
    main()
