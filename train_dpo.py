"""DPO (Direct Preference Optimization) sobre o adaptador LoRA já treinado (SFT), em PyTorch puro.

A ideia: para cada pedido temos uma versão boa (chosen) e uma ruim (rejected). O DPO ajusta o adaptador para aumentar
a probabilidade da boa em relação à ruim, sem se afastar demais do modelo SFT (que serve de referência, congelado):

    perda = -log sigmoid( beta * [ (logp_chosen - logp_rejected)  -  (ref_chosen - ref_rejected) ] )

O treino continua o MESMO adaptador (SFT), então o resultado é um único adaptador novo em adapter_dpo/<lang>.
As probabilidades da referência são calculadas uma vez, antes do treino, com o adaptador SFT ainda intacto.

    python train_dpo.py --lang pt          # le data/pairs.jsonl, le adapter/pt, grava adapter_dpo/pt
"""
import argparse
import json
import math
import os
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from common import ADAPTER_DIR, BASES, adapter_path


def encode(tok, prompt, completion, finished, max_len):
    p = tok(prompt)["input_ids"]
    c = tok(completion)["input_ids"] + ([tok.eos_token_id] if finished else [])  # só fecha com EOS se terminou de fato
    return (p + c)[:max_len], ([-100] * len(p) + c)[:max_len]


def seq_logps(model, seqs, device):
    """Soma do log-prob dos tokens da resposta (não do pedido) para cada sequência (ids, labels)."""
    n = max(len(i) for i, _ in seqs)
    ids = torch.tensor([i + [0] * (n - len(i)) for i, _ in seqs], device=device)
    att = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in seqs], device=device)
    lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in seqs], device=device)[:, 1:]
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    mask = lab != -100
    tok_lp = logits.gather(2, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return (tok_lp * mask).sum(-1)


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def pair_seqs(tok, rows, max_len):
    ch = [encode(tok, r["prompt"], r["chosen"], r["chosen_finished"], max_len) for r in rows]
    rj = [encode(tok, r["prompt"], r["rejected"], r["rejected_finished"], max_len) for r in rows]
    return ch, rj


@torch.no_grad()
def ref_margins(model, ch, rj, device, bs=8):
    out = []
    for i in range(0, len(ch), bs):
        c, r = ch[i:i + bs], rj[i:i + bs]
        lp = seq_logps(model, c + r, device)
        out += (lp[:len(c)] - lp[len(c):]).tolist()
    return out  # (logp_chosen - logp_rejected) do modelo de referência, por par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(BASES))
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--sft-root", default=ADAPTER_DIR, help="onde está o adaptador SFT (adapter/<lang>)")
    ap.add_argument("--out", default="adapter_dpo", help="grava <out>/<lang>")
    ap.add_argument("--results", default="results")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="1e-5 = toque leve (na 1ª tentativa, 5e-5 x 3 épocas memorizou os pares e perdeu diversidade)")
    ap.add_argument("--epochs", type=int, default=1, help="a validação já estagnava na época 1")
    ap.add_argument("--batch", type=int, default=4, help="pares por passo de GPU")
    ap.add_argument("--accum", type=int, default=2, help="acúmulo de gradiente")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--val", type=float, default=0.1, help="fração dos pares só para validar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=8, help="mínimo de pares para treinar")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    rows = [r for r in rows if r["lang"] == a.lang]
    if len(rows) < a.min_pairs:
        raise SystemExit(f"Poucos pares em '{a.lang}' ({len(rows)}). Rode make_pairs.py com --per-cell maior.")
    random.shuffle(rows)
    n_val = int(len(rows) * a.val) if len(rows) >= 40 else 0
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"[{a.lang}] pares: treino={len(train_rows)} validação={len(val_rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = BASES[a.lang]
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(model, adapter_path(a.lang, a.sft_root), is_trainable=True).to(device)
    model.eval()  # sem dropout: política e referência comparáveis

    tr_c, tr_r = pair_seqs(tok, train_rows, a.max_len)
    va_c, va_r = pair_seqs(tok, val_rows, a.max_len)
    print("calculando probabilidades da referência (SFT congelado)...")
    tr_ref = ref_margins(model, tr_c, tr_r, device)
    va_ref = ref_margins(model, va_c, va_r, device) if val_rows else []

    @torch.no_grad()
    def evaluate_val():
        if not val_rows:
            return None
        m = ref_margins(model, va_c, va_r, device)
        d = [a.beta * (x - y) for x, y in zip(m, va_ref)]
        return {"acc": sum(v > 0 for v in d) / len(d), "margem": sum(d) / len(d)}

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"parâmetros treináveis: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train_rows) / (a.batch * a.accum)) * a.epochs
    sched = get_scheduler("linear", opt, num_warmup_steps=max(1, int(0.1 * steps)), num_training_steps=steps)

    log, step = [], 0
    for epoch in range(a.epochs):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        acc_loss = acc_hit = acc_mar = acc_n = 0
        for k, idx in enumerate(chunks(order, a.batch), 1):
            c, r = [tr_c[i] for i in idx], [tr_r[i] for i in idx]
            lp = seq_logps(model, c + r, device)
            pol = lp[:len(c)] - lp[len(c):]
            ref = torch.tensor([tr_ref[i] for i in idx], device=device)
            margin = a.beta * (pol - ref)
            loss = -torch.nn.functional.logsigmoid(margin).mean()
            (loss / a.accum).backward()
            acc_loss += loss.item() * len(idx)
            acc_hit += (margin > 0).sum().item()
            acc_mar += margin.sum().item()
            acc_n += len(idx)
            if k % a.accum == 0 or k * a.batch >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0 or step == steps:
                    rec = {"passo": step, "epoca": epoch + 1, "loss": acc_loss / acc_n,
                           "acc": acc_hit / acc_n, "margem": acc_mar / acc_n}
                    log.append(rec)
                    print(f"passo {step}/{steps} | loss {rec['loss']:.4f} | acerto {rec['acc']:.0%} | margem {rec['margem']:+.3f}")
                    acc_loss = acc_hit = acc_mar = acc_n = 0
        v = evaluate_val()
        if v:
            print(f"== fim da época {epoch + 1} | validação: acerto {v['acc']:.0%} | margem {v['margem']:+.3f}")
            log.append({"epoca": epoch + 1, "validacao": v})

    out = adapter_path(a.lang, a.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, f"dpo_metrics_{a.lang}.json"), "w") as f:
        json.dump({"base": base, "args": vars(a), "n_treino": len(train_rows), "n_val": len(val_rows), "log": log}, f, indent=2)
    print(f"adaptador DPO salvo em {out}/")


if __name__ == "__main__":
    main()
