"""Playground generativo de fantasia para RPG (Streamlit)."""
import os
import time

import streamlit as st

import engine
from checks import blocked_in, check_request, repair_occupations
from common import ADAPTER_DIR, BASES, EXAMPLES, LANGS, MODES

st.set_page_config(page_title="Forja de Contos — RPG", page_icon="🐉", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner="Carregando o modelo do idioma escolhido...", max_entries=1)
def get_model(lang: str):
    # max_entries=1: ao trocar de idioma o modelo anterior sai da memória (importante no deploy gratuito)
    return engine.load_lang(lang, os.path.join(HERE, ADAPTER_DIR))


st.title("🐉 Forja de Contos")
st.caption("GPT-2 ajustado com LoRA para gerar contos, cenários, NPCs, famílias de NPCs, missões, rumores e poemas (em PT) de RPG de fantasia. "
           "Um modelo para cada idioma: português e inglês.")

with st.sidebar:
    st.header("Configuração")
    lang = st.radio("Idioma / Language", list(LANGS), format_func=LANGS.get, horizontal=True)
    # alguns modos existem só em um idioma (poema: só português)
    mode = st.selectbox("Modo", [m for m in MODES if lang in MODES[m]], format_func=lambda m: MODES[m][lang])

    model, tok = get_model(lang)
    tuned = engine.has_adapter(model)
    st.caption(f"Base: `{BASES[lang]}`")
    has_dpo = "dpo" in engine.adapter_names(model)
    if tuned:
        use_adapter = st.toggle("Usar modelo ajustado (LoRA)", value=True,
                                help="Desligue para ver o modelo base original e comparar.")
        version = "default"
        if has_dpo and use_adapter:
            version = "dpo" if st.radio("Versão do ajuste", ["SFT", "SFT + DPO"], horizontal=True,
                                        help="DPO: ajuste extra por preferência, para reduzir erros de formato.") == "SFT + DPO" else "default"
    else:
        use_adapter = False
        version = "default"
        st.warning("Adaptador não encontrado — usando o modelo base original.")

    st.subheader("Parâmetros de geração")
    n = st.slider("Quantidade de versões", 1, 4, 1)
    max_new_tokens = st.slider("Tamanho máximo (tokens)", 50, 900, 450, 10,
                               help="Só um teto: o modelo ajustado para sozinho. A família de NPCs usa até ~360.")
    temperature = st.slider("Temperatura", 0.1, 1.5, 0.7, 0.05,
                            help="Mais alta = mais criativo, porém mais erros (nomes trocados, estrutura quebrada).")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.92, 0.01)
    top_k = st.slider("Top-k", 0, 200, 50, 5)
    # poema: 1,15 evita os loops de refrão ("Sou o céu!" repetido); nos demais modos 1,05 preserva os nomes do pedido
    repetition_penalty = st.slider("Penalidade de repetição", 1.0, 1.5, 1.15 if mode == "poema" else 1.05, 0.01,
                                   key=f"rep_{mode}",
                                   help="Valores altos penalizam até os nomes copiados do pedido.")
    no_repeat = st.slider("Bloquear n-gramas repetidos", 0, 5, 0, help="0 = desligado.")
    seed_on = st.checkbox("Semente fixa")
    seed = st.number_input("Semente", value=42, step=1, disabled=not seed_on)
    verify = st.toggle("Conferir o formato e tentar de novo", value=True,
                       help="Depois de gerar, confere a estrutura do modo (campos, nº de NPCs, nomes repetidos, término) "
                            "com a mesma checagem da avaliação e gera de novo, até 4 vezes, se falhar. Poema não tem checagem.")

request = st.text_area("Pedido", value=EXAMPLES[lang][mode], height=90,
                       help="Dica: siga o estilo do exemplo para melhores resultados.")
if mode == "poema":
    st.caption("Poema de estilo livre: o modelo aprendeu a forma do poema, e o título só o inspira (não garante o assunto).")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("✨ Gerar", type="primary", disabled=not request.strip()):
    bad = blocked_in(request)
    if bad:
        st.error("Este pedido contém palavras que o app não gera (" + ", ".join(bad) + "). Reescreva o pedido.")
        st.stop()
    t0 = time.time()
    def gen(k, attempt):
        return engine.generate(
            model, tok, lang, mode, request, use_adapter=use_adapter, adapter=version if use_adapter else None, n=k,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat,
            seed=int(seed) + attempt if seed_on else None,
        )

    def fix(o):  # ocupação inventada -> palavra do pedido que o modelo tentava escrever (avisado na legenda)
        if mode in ("npc", "familia") and use_adapter:
            o["text"], n_fix = repair_occupations(lang, request, o["text"])
            return n_fix
        return 0

    with st.spinner("Escrevendo..."):
        outs = gen(n, 0)
        for o in outs:
            o["fixed"] = fix(o)
            o["tries"], o["issues"] = 1, check_request(lang, mode, request, o["text"], o["finished"])
        if verify and use_adapter:
            for o in outs:  # só as versões que falharam são geradas de novo
                while o["issues"] and o["tries"] < 4:
                    new = gen(1, o["tries"])[0]
                    tries = o["tries"] + 1
                    new["fixed"] = fix(new)
                    o.update(new, tries=tries, issues=check_request(lang, mode, request, new["text"], new["finished"]))
    st.session_state.history.insert(0, {
        "lang": lang, "mode": mode, "request": request, "outs": outs, "dt": time.time() - t0,
        "model": ("DPO" if version == "dpo" else "ajustado") if use_adapter else "base",
    })

for i, h in enumerate(st.session_state.history):
    label = f"[{h['lang']}] {MODES[h['mode']][h['lang']]} · {h['model']} · {h['request'][:60]}"
    with st.expander(label, expanded=(i == 0)):
        cols = st.columns(len(h["outs"]))
        for c, o in zip(cols, h["outs"]):
            c.markdown(o["text"].replace("\n", "  \n"))
            if not o["finished"]:
                c.caption("⚠️ cortado no limite de tokens")
            if o.get("fixed"):
                c.caption(f"🔧 {o['fixed']} ocupação(ões) escrita(s) errada(s) corrigida(s) com a palavra do pedido")
            if o.get("issues") is not None:  # None = modo sem checagem (poema)
                tries = f" (tentativa {o['tries']})" if o["tries"] > 1 else ""
                c.caption(f"✅ formato conferido{tries}" if not o["issues"] else "⚠️ falhou na checagem: " + "; ".join(o["issues"]))
        st.caption(f"{h['dt']:.1f}s")

if st.session_state.history and st.sidebar.button("Limpar histórico"):
    st.session_state.history = []
    st.rerun()
