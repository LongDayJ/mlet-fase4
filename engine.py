"""Carregamento do modelo (base + adaptador LoRA) e geração de texto."""
import contextlib
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ADAPTER_DIR, BASES, BLOCKED_WORDS, DPO_DIR, adapter_path, build_prompt


def load_lang(lang: str, root: str = ADAPTER_DIR, dpo_root: str = DPO_DIR):
    """Carrega a base do idioma + o adaptador SFT (adapter/<lang>, nome 'default') e, se existir, o DPO ('dpo')."""
    model, tok = load(BASES[lang], adapter_path(lang, root))
    dpo = adapter_path(lang, dpo_root)
    if has_adapter(model) and os.path.isdir(dpo):
        model.load_adapter(dpo, adapter_name="dpo")
        model.set_adapter("default")
    return model, tok


def adapter_names(model) -> list[str]:
    """Nomes dos adaptadores carregados: 'default' (SFT) e, se houver, 'dpo'."""
    return list(model.peft_config.keys()) if hasattr(model, "peft_config") else []


@contextlib.contextmanager
def using(model, adapter: str | None):
    """Ativa um adaptador pelo nome durante o bloco; None = modelo base puro (adaptadores desligados)."""
    if not has_adapter(model):
        yield
    elif adapter is None:
        with model.disable_adapter():
            yield
    else:
        prev = model.active_adapter
        prev = prev[0] if isinstance(prev, (list, tuple)) else prev
        model.set_adapter(adapter)
        try:
            yield
        finally:
            model.set_adapter(prev)


def load(base: str, adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base)
    if adapter and os.path.isdir(adapter):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # no Colab usa a GPU; no deploy/PC fica na CPU
    model.eval()
    return model, tok


def has_adapter(model) -> bool:
    return hasattr(model, "disable_adapter")


_BAD_CACHE: dict = {}


def _bad_ids(tok):
    """Sequências de tokens das palavras bloqueadas (com/sem espaço, minúscula/Capitalizada/MAIÚSCULA)."""
    key = id(tok)
    if key not in _BAD_CACHE:
        ids = set()
        for w in BLOCKED_WORDS:
            # minúscula só com espaço antes (início de palavra): sem espaço bateria em pedaços de palavras
            # comuns ("rape" em "drapejar"); Capitalizada/MAIÚSCULA também valem no início de linha
            ids.add(tuple(tok(" " + w, add_special_tokens=False)["input_ids"]))
            for v in {w.capitalize(), w.upper()}:
                for pre in ("", " "):
                    ids.add(tuple(tok(pre + v, add_special_tokens=False)["input_ids"]))
        _BAD_CACHE[key] = [list(t) for t in ids if t]
    return _BAD_CACHE[key]


@torch.inference_mode()
def generate(
    model,
    tok,
    lang: str,
    mode: str,
    request: str,
    use_adapter: bool = True,
    n: int = 1,
    max_new_tokens: int = 450,  # família de NPCs chega a ~360 tokens; os outros modos param sozinhos antes
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.05,  # >1.1 atrapalha: penaliza também os tokens do pedido (nomes de lugares etc.)
    no_repeat_ngram_size: int = 0,
    seed: int | None = None,
    adapter: str | None = None,
):
    """Retorna lista de {"text": str, "finished": bool} (finished = parou sozinho com EOS).
    adapter: 'default' (SFT) ou 'dpo'; se None vale use_adapter (True = SFT, False = base puro)."""
    if seed is not None:
        torch.manual_seed(seed)
    prompt = build_prompt(lang, mode, request)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_pos = getattr(model.config, "n_positions", None)  # GPT-2 original: 1024; alguns modelos: 256
    if n_pos:
        max_new_tokens = max(1, min(max_new_tokens, n_pos - enc["input_ids"].shape[1]))
    bad_ids = _bad_ids(tok)
    chosen = adapter if adapter is not None else ("default" if use_adapter else None)
    with using(model, chosen):
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            bad_words_ids=bad_ids,
        )
    results = []
    start = enc["input_ids"].shape[1]
    for o in out:
        ids = o[start:].tolist()
        finished = tok.eos_token_id in ids
        if finished:
            ids = ids[: ids.index(tok.eos_token_id)]
        results.append({"text": tok.decode(ids, skip_special_tokens=True).strip(), "finished": finished})
    return results
