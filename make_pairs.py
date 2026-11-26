"""Gera pares de preferência (versão boa x versão ruim) para o DPO, usando o modelo ajustado (SFT).

Para cada pedido o modelo escreve K versões em temperatura alta (para provocar falhas); o checks.py marca quais passam
(terminou sozinha, estrutura do modo, cita lugar/herói do pedido, sem linhas duplicadas). Cada par é
(versão que passa = chosen, versão que falha = rejected) para o MESMO pedido. Pedidos sem os dois tipos são descartados.
Parte dos pedidos usa lugares inventados (fora do dataset), para ensinar a copiar o nome do pedido.

    python make_pairs.py --target-pairs 20 --max-prompts 120 --k 8   # equilibrado: ~20 pares por idioma e modo
    python make_pairs.py --target-pairs 20 --resume                  # retoma depois de uma queda (conta os pares já gravados)
    python make_pairs.py --per-cell 50 --k 8     # alternativa: 50 pedidos fixos por célula (modos fáceis geram poucos pares)
    python make_pairs.py --per-cell 2 --k 4      # teste rápido
"""
import argparse
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "data"))
import build_dataset as bd  # noqa: E402

import engine  # noqa: E402
from checks import problems, requirements  # noqa: E402
from common import ADAPTER_DIR, BASES, build_prompt  # noqa: E402

NOVEL_PLACES = {
    "pt": [("f", "Fortaleza de Cristal"), ("m", "Reino das Brumas"), ("f", "Ilha da Lua Cega"),
           ("m", "Santuário de Sal"), ("f", "Mina dos Anões Perdidos"), ("m", "Jardim das Estátuas"),
           ("f", "Estalagem do Dragão Cego"), ("m", "Pico do Trovão")],
    "en": [("f", "Crystal Fortress"), ("m", "Realm of Mists"), ("f", "Isle of the Blind Moon"),
           ("m", "Salt Sanctuary"), ("f", "Mine of the Lost Dwarves"), ("m", "Garden of Statues"),
           ("f", "Inn of the Blind Dragon"), ("m", "Thunder Peak")],
}


def make_request(lang, mode, rng, novel_p):
    """Pedido no mesmo formato do dataset. Devolve (pedido, lugar, herói)."""
    P = dict(bd.PACKS[lang])
    if rng.random() < novel_p:
        P["places"] = NOVEL_PLACES[lang]
    c = bd.ctx(P, rng)
    place = c["pl"]["the"].split(" ", 1)[1]
    tm, tf, _ = c["tone"]
    vowel = tm[:1].lower() in "aeiou"
    if mode == "conto":
        req = rng.choice(P["req_conto"]).format(tm=tm, tf=tf, a="an" if vowel else "a", A="An" if vowel else "A", **c)
    elif mode == "cenario":
        req = rng.choice(P["req_cen"]).format(**c)
    elif mode == "npc":
        role = rng.choice(P["roles"])[0 if c["g"] == "m" else 1]
        req = rng.choice(P["req_npc"]).format(role=role, **c)
    elif mode == "missao":
        req = rng.choice(P["req_quest"]).format(lvl=rng.randint(1, 10), **c)
    elif mode == "familia":
        req, members, _kind = bd.fam_request(P, rng, c)
        return req, place, c["h"], members
    else:
        req = rng.choice(P["req_rumor"]).format(**c)
    return req, place, c["h"], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(BASES), choices=list(BASES))
    ap.add_argument("--adapter-root", default=ADAPTER_DIR)
    ap.add_argument("--per-cell", type=int, default=50, help="pedidos por (idioma, modo)")
    ap.add_argument("--k", type=int, default=8, help="versões geradas por pedido")
    ap.add_argument("--temp", type=float, default=1.3, help="temperatura alta para provocar falhas (o SFT erra pouco)")
    ap.add_argument("--novel", type=float, default=0.3, help="fração de pedidos com lugar inventado")
    ap.add_argument("--max-new-tokens", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--target-pairs", type=int, default=0,
                    help="pares por (idioma, modo); >0 equilibra: continua sorteando pedidos até atingir (ou --max-prompts)")
    ap.add_argument("--max-prompts", type=int, default=120, help="teto de pedidos por célula quando --target-pairs > 0")
    ap.add_argument("--resume", action="store_true", help="continua um --out existente (conta os pares já gravados)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    have = collections.Counter()  # pares já gravados por (idioma, modo), para --resume
    if a.resume and pathlib.Path(a.out).exists():
        for line in open(a.out, encoding="utf-8"):
            r = json.loads(line)
            have[(r["lang"], r["mode"])] += 1
        rng = random.Random(a.seed + sum(have.values()))  # evita repetir os mesmos pedidos ao retomar
    stats = collections.defaultdict(lambda: [0, 0, 0, 0])  # pedidos, amostras ok, amostras, pares
    reasons = collections.Counter()
    with open(a.out, "a" if a.resume else "w", encoding="utf-8") as fh:
        for lang in a.langs:
            model, tok = engine.load_lang(lang, a.adapter_root)
            for mode in bd.GENS:
                s = stats[(lang, mode)]
                s[3] = have[(lang, mode)]
                limit = a.max_prompts if a.target_pairs else a.per_cell
                tried = 0
                while tried < limit and not (a.target_pairs and s[3] >= a.target_pairs):
                    tried += 1
                    req, place, hero, members = make_request(lang, mode, rng, a.novel)
                    required, numbered = requirements(lang, mode, req, place, hero)
                    outs = engine.generate(model, tok, lang, mode, req, n=a.k, max_new_tokens=a.max_new_tokens,
                                           temperature=a.temp, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                           seed=rng.randrange(1 << 30))
                    good, bad = [], []
                    for o in outs:
                        p = problems(o["text"], o["finished"], required, numbered, members)
                        (bad if p else good).append((o, p))
                    s[0] += 1
                    s[1] += len(good)
                    s[2] += len(outs)
                    for _o, p in bad:
                        for x in p:
                            reasons[x.split("'")[0].strip() or x] += 1
                    if good and bad:
                        g, (b, why) = rng.choice(good)[0], rng.choice(bad)
                        row = {"lang": lang, "mode": mode, "request": req,
                               "prompt": build_prompt(lang, mode, req),
                               "chosen": g["text"], "chosen_finished": True,
                               "rejected": b["text"], "rejected_finished": b["finished"], "reason": why}
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        s[3] += 1
            del model

    print(f"\npares gravados em {a.out}\n")
    print(f"{'idioma/modo':16s} {'pedidos':>8s} {'taxa ok':>8s} {'pares':>6s}")
    for (lang, mode), (n, ok, tot, pairs) in stats.items():
        print(f"{lang + '/' + mode:16s} {n:8d} {ok / max(1, tot):8.0%} {pairs:6d}")
    print("total de pares:", sum(v[3] for v in stats.values()))
    print("motivos de falha:", dict(reasons.most_common(6)))


if __name__ == "__main__":
    main()
