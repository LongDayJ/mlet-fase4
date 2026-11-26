"""Avalia, por idioma, o modelo base vs. o ajustado: perplexidade, diversidade, repetição e taxa de término.

    python evaluate.py                # compara base e ajustado em pt e en
    python evaluate.py --sweep        # também varre temperature/penalidade no modelo ajustado
    python evaluate.py --quick        # teste rápido (poucos exemplos, textos curtos)

Saídas em results/: eval.md (tabelas), eval.json e samples.md (textos lado a lado, bom para o vídeo).
"""
import argparse
import collections
import json
import math
import os

import torch
import torch.nn.functional as F

import engine
from checks import CASES, problems
from common import ADAPTER_DIR, BASES, DPO_DIR, EXAMPLES, MODES, encode_example, make_collate

GEN_KW = dict(max_new_tokens=450, temperature=0.7, top_p=0.92, top_k=50, repetition_penalty=1.05)


def ngrams(words, n):
    return list(zip(*[words[i:] for i in range(n)]))


def distinct(texts, n):
    grams = [g for t in texts for g in ngrams(t.split(), n)]
    return len(set(grams)) / max(1, len(grams))


def repetition(texts):
    """Média da fração de trigramas repetidos dentro de cada texto (0 = nenhum, 1 = só repetição)."""
    vals = []
    for t in texts:
        g = ngrams(t.split(), 3)
        if g:
            vals.append(1 - len(set(g)) / len(g))
    return sum(vals) / max(1, len(vals))


def gen_args(adapter):
    """adapter: None = modelo base puro; 'default' = SFT; 'dpo' = depois do DPO."""
    return {"adapter": adapter} if adapter else {"use_adapter": False}


@torch.inference_mode()
def perplexity(model, tok, rows, adapter, bs=8):
    """Devolve (perplexidade geral, {modo: perplexidade}). Por modo porque os textos naturais (poema) são muito menos
    previsíveis que os feitos por moldes e puxam a média geral para cima."""
    col = make_collate(tok)
    acc = collections.defaultdict(lambda: [0.0, 0])  # chave -> [soma da perda, nº de tokens]
    with engine.using(model, adapter):
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            batch = col([encode_example(tok, r["prompt"], r["completion"]) for r in chunk])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            lg, lb = logits[:, :-1], batch["labels"][:, 1:]
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100,
                                 reduction="none").view(lb.shape)
            mask = lb != -100
            for r, loss, n in zip(chunk, (ce * mask).sum(1).tolist(), mask.sum(1).tolist()):
                for key in (r["mode"], "_geral"):
                    acc[key][0] += loss
                    acc[key][1] += n
    geral = math.exp(acc["_geral"][0] / acc["_geral"][1])
    return geral, {m: math.exp(v[0] / v[1]) for m, v in acc.items() if m != "_geral"}


def sample_all(model, tok, lang, adapter, n, kw):
    out = []
    for mode, req in EXAMPLES[lang].items():
        for s in engine.generate(model, tok, lang, mode, req, n=n, seed=0, **gen_args(adapter), **kw):
            out.append({"lang": lang, "mode": mode, "request": req, **s})
    return out


def metrics(samples):
    texts = [s["text"] for s in samples]
    return {
        "distinct-1": distinct(texts, 1), "distinct-2": distinct(texts, 2), "distinct-3": distinct(texts, 3),
        "repeticao_trigramas": repetition(texts),
        "palavras_medias": sum(len(t.split()) for t in texts) / max(1, len(texts)),
        "terminou_sozinho": sum(s["finished"] for s in samples) / max(1, len(samples)),
    }


def format_eval(model, tok, lang, adapter, n, kw):
    """Fração de amostras 'ok' (ver checks.py) por modo, para o pedido 'visto' e o de lugar 'novo'."""
    out = {}
    for mode, cases in CASES[lang].items():
        for kind, case in zip(("visto", "novo"), cases):
            req, required, numbered, *rest = case  # 4º item (família): nº de NPCs; 5º (poema): mínimo de versos
            members = rest[0] if rest else 0
            min_lines = rest[1] if len(rest) > 1 else 0
            samples = engine.generate(model, tok, lang, mode, req, n=n, seed=0, **gen_args(adapter), **kw)
            ok = sum(not problems(s["text"], s["finished"], required, numbered, members, min_lines) for s in samples)
            out[f"{mode}/{kind}"] = ok / len(samples)
    return out


def format_table(fmt):
    keys = []
    for m in fmt.values():  # união das colunas: alguns modos existem só em um idioma (poema: só PT)
        keys += [k for k in m if k not in keys]
    lines = ["| config | " + " | ".join(keys) + " | média |", "|---|" + "---|" * (len(keys) + 1)]
    for name, m in fmt.items():
        cells = " | ".join(f"{m[k]:.0%}" if k in m else "—" for k in keys)
        lines.append(f"| {name} | {cells} | {sum(m.values()) / len(m):.0%} |")
    return "\n".join(lines)


def ppl_modes_table(pm):
    modes = []
    for m in pm.values():
        modes += [k for k in m if k not in modes]
    lines = ["| modo | " + " | ".join(pm) + " |", "|---|" + "---|" * len(pm)]
    for mode in modes:
        lines.append(f"| {mode} | " + " | ".join(f"{m[mode]:.2f}" if mode in m else "—" for m in pm.values()) + " |")
    return "\n".join(lines)


def table(rows):
    keys = list(next(iter(rows.values())).keys())
    lines = ["| config | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.3f}" for k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(BASES), choices=list(BASES))
    ap.add_argument("--adapter-root", default=ADAPTER_DIR)
    ap.add_argument("--dpo-root", default=DPO_DIR, help="se adapter_dpo/<lang> existir, entra na comparação")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-val", type=int, default=200, help="exemplos de validação por idioma")
    ap.add_argument("--n", type=int, default=3, help="amostras por prompt")
    ap.add_argument("--n-format", type=int, default=8,
                    help="amostras por pedido na checagem de formato (use 32 para comparar SFT x DPO com menos ruído)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    kw = dict(GEN_KW)
    if a.quick:
        a.n, a.n_format, a.max_val, a.sweep = 1, 1, 16, False
        kw["max_new_tokens"] = 60

    all_val = [json.loads(l) for l in open(a.val, encoding="utf-8")]
    ppl, ppl_modes, results, sweep, gens, fmt = {}, {}, {}, {}, {}, {}

    for lang in a.langs:
        model, tok = engine.load_lang(lang, a.adapter_root, a.dpo_root)
        names = engine.adapter_names(model)
        tuned = "default" in names
        if not tuned:
            print(f"[{lang}] adaptador não encontrado em {a.adapter_root}/{lang}: avaliando só o base")
        val = [r for r in all_val if r["lang"] == lang][: a.max_val]

        # base puro, SFT ('ajustado') e, se existir adapter_dpo/<lang>, o modelo depois do DPO
        variants = [("base", None)] + ([("ajustado", "default")] if tuned else []) + \
                   ([("dpo", "dpo")] if "dpo" in names else [])
        for name, adapter in variants:
            key = f"{lang} · {name}"
            ppl[key], ppl_modes[key] = perplexity(model, tok, val, adapter)
            gens[key] = sample_all(model, tok, lang, adapter, a.n, kw)
            results[key] = metrics(gens[key])
            fmt[key] = format_eval(model, tok, lang, adapter, a.n_format, kw)
        print(f"[{lang}] perplexidade:", {k: round(v, 2) for k, v in ppl.items() if k.startswith(lang)})

        if a.sweep and tuned:
            for t in (0.5, 0.7, 0.9, 1.1):
                for rp in (1.0, 1.05, 1.1):
                    sweep[f"{lang} · T={t} rep={rp}"] = metrics(
                        sample_all(model, tok, lang, "default", a.n, {**kw, "temperature": t, "repetition_penalty": rp}))
        del model

    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "eval.json"), "w", encoding="utf-8") as f:
        json.dump({"bases": {l: BASES[l] for l in a.langs}, "perplexity": ppl, "perplexidade_por_modo": ppl_modes,
                   "metrics": results, "formato_correto": fmt, "sweep": sweep}, f, indent=2, ensure_ascii=False)

    md = ["# Avaliação\n", "Bases: " + ", ".join(f"`{l}` = {BASES[l]}" for l in a.langs) + "\n",
          "## Perplexidade (validação, só na parte gerada)\n",
          "| modelo | perplexidade |\n|---|---|", *[f"| {k} | {v:.2f} |" for k, v in ppl.items()],
          "\n### Perplexidade por modo\n",
          "Textos de moldes ficam perto de 1; texto natural (poema) é muito menos previsível e puxa a média geral para cima.\n",
          ppl_modes_table(ppl_modes),
          "\n## Formato correto (% de amostras ok)\n",
          "Ok = terminou sozinha, tem a estrutura do modo, cita o lugar/personagem do pedido, sem linhas duplicadas. "
          "`visto` = lugar que existe no dataset; `novo` = lugar inventado (mede se copia o nome do pedido).\n",
          format_table(fmt),
          "\n## Métricas de geração (diversidade etc.)\n", table(results)]
    if sweep:
        md += ["\n## Varredura de parâmetros (modelo ajustado)\n", table(sweep)]
    with open(os.path.join(a.results, "eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    sm = ["# Amostras lado a lado\n"]
    for lang in a.langs:
        base = gens.get(f"{lang} · base", [])
        others = [(n, gens[f"{lang} · {n}"]) for n in ("ajustado", "dpo") if f"{lang} · {n}" in gens]
        for i in range(0, len(base), a.n):
            s = base[i]
            sm.append(f"## [{lang}] {MODES[s['mode']][lang]} — {s['request']}\n")
            sm.append(f"**base**\n\n{s['text']}\n")
            for name, g in others:
                sm.append(f"**{name}**\n\n{g[i]['text']}\n")
    with open(os.path.join(a.results, "samples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(sm))
    print("\n".join(md))


if __name__ == "__main__":
    main()
