"""Formato de prompt e utilidades compartilhadas por dataset, treino, avaliação e app."""

import os

# Um modelo base e um adaptador LoRA por idioma. Dá para trocar sem mexer no código:
#   BASE_PT=yanH2003/fine-tuned-gpt2-portuguese   BASE_EN=gpt2
BASES = {
    "pt": os.environ.get("BASE_PT", "pierreguillou/gpt2-small-portuguese"),
    "en": os.environ.get("BASE_EN", "gpt2"),
}
ADAPTER_DIR = "adapter"  # adapter/pt e adapter/en (SFT)
DPO_DIR = "adapter_dpo"  # adapter_dpo/pt e adapter_dpo/en (depois do DPO; opcional)


def adapter_path(lang: str, root: str = ADAPTER_DIR) -> str:
    return os.path.join(root, lang)

LANGS = {"pt": "Português", "en": "English"}

MODES = {
    "conto": {"pt": "Conto curto", "en": "Short tale"},
    "cenario": {"pt": "Cenário / local", "en": "Scene / location"},
    "npc": {"pt": "NPC", "en": "NPC"},
    "missao": {"pt": "Missão", "en": "Quest"},
    "rumor": {"pt": "Rumores de taverna", "en": "Tavern rumors"},
    # família: PT acerta ~96%/75% (pedido visto/novo), EN ~75%/79% (24 amostras por pedido: margem de ±9 pontos)
    "familia": {"pt": "Família de NPCs (2 a 4)", "en": "NPC family (2 to 4)"},
    "poema": {"pt": "Poema / canção de bardo"},  # só em português (dataset de poemas em PT)
}

EXAMPLES = {
    "pt": {
        "conto": "Um conto sombrio sobre Lysa, a maga, na Torre do Corvo.",
        "cenario": "Descreva o Pântano de Vidro.",
        "npc": "Crie um NPC: ferreira no Porto de Âmbar.",
        "missao": "Crie uma missão para um grupo de nível 3 na Cidadela de Ferro.",
        "rumor": "Quais são os rumores na taverna sobre a Ilha dos Faróis Mortos?",
        "familia": "Crie uma família de três NPCs: ferreiros no Porto de Âmbar.",
        "poema": 'Escreva um poema chamado "A Torre do Corvo".',
    },
    "en": {
        "conto": "A grim tale about Lysa, the mage, in the Raven Tower.",
        "cenario": "Describe the Glass Marsh.",
        "npc": "Create an NPC: blacksmith in the Amber Harbor.",
        "missao": "Create a quest for a level 3 party in the Iron Citadel.",
        "rumor": "What are the tavern rumors about the Isle of Dead Lighthouses?",
        "familia": "Create a family of three NPCs: blacksmiths in the Amber Harbor.",
    },
}


def build_prompt(lang: str, mode: str, request: str) -> str:
    # Rótulos em inglês nos dois idiomas: um formato só, e o GPT-2 base não confunde o idioma do texto
    return f"### Language: {lang}\n### Mode: {mode}\n### Request: {request.strip()}\n### Text:\n"


def encode_example(tok, prompt: str, completion: str, max_len: int = 512):
    """Tokeniza prompt+completion+EOS. A perda só conta na completion (e no EOS, para o modelo aprender a parar)."""
    p = tok(prompt)["input_ids"]
    c = tok(completion)["input_ids"] + [tok.eos_token_id]
    return {
        "input_ids": (p + c)[:max_len],
        "labels": ([-100] * len(p) + c)[:max_len],
    }


def make_collate(tok):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, att, lab = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            att.append([1] * len(b["input_ids"]) + [0] * pad)
            lab.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(att),
            "labels": torch.tensor(lab),
        }

    return collate


# Palavras que o app não deve gerar. O modelo as monta por semelhança de sílabas (ex.: "pedreiro" -> "pedófilo").
# Base: lista pública LDNOOBW (PT e EN), filtrada à mão: ficaram abuso sexual, sexo explícito, xingamentos pesados e
# ofensas a grupos; saíram palavras comuns em fantasia ou ambíguas (veado = cervo, pinto = filhote, cock, dick, hell...).
BLOCKED_PT = [
    # abuso sexual / crianças
    "pedófilo", "pedófila", "pedófilos", "pedofilo", "pedofilia", "estupro", "estupros", "estuprador", "estupradora",
    "estupradores", "estuprar", "estuprou", "abuso sexual", "incesto", "zoofilia", "necrofilia", "lolita", "loli", "shota",
    # sexo explícito
    "pornografia", "porno", "pornô", "boceta", "buceta", "xoxota", "xana", "xereca", "caralho", "piroca", "punheta",
    "siririca", "boquete", "orgia", "esporra", "masturbação", "ejaculação",
    # xingamentos pesados
    "porra", "foder", "foda-se", "fodido", "puta", "puta que pariu", "filho da puta", "vadia", "vagabunda", "arrombado",
    "cuzão", "merda",
    # ofensas a grupos
    "viado", "paneleiro", "sapatão", "traveco", "baitola", "crioulo", "retardado", "mongoloide", "heil hitler",
]
BLOCKED_EN = [
    # sexual abuse / minors
    "pedophile", "pedophiles", "paedophile", "pedophilia", "paedophilia", "child porn", "pthc", "jailbait", "lolicon",
    "loli", "shota", "rape", "raped", "raping", "rapist", "rapists", "date rape", "molest", "molester", "molested",
    "incest", "bestiality", "zoophilia", "necrophilia",
    # explicit sex
    "porn", "porno", "pornography", "hentai", "blowjob", "handjob", "cumshot", "jizz", "cunt", "pussy", "dildo",
    "orgasm", "orgy", "gangbang", "masturbate", "masturbation", "anal sex", "wank", "wanker",
    # heavy profanity
    "fuck", "fucked", "fucker", "fucking", "motherfucker", "shit", "bullshit", "bitch", "whore", "slut", "asshole",
    # slurs
    "nigger", "nigga", "faggot", "kike", "spic", "gook", "wetback", "beaner", "towelhead", "raghead", "tranny",
    "retard", "retarded", "dyke", "shemale", "paki", "white power", "heil hitler",
]
BLOCKED_WORDS = BLOCKED_PT + BLOCKED_EN

# Radicais (início de palavra): bloqueiam todas as flexões, listadas ou não. Só os mais graves, para não pegar
# palavras inocentes (ex.: "rape" sozinho pegaria "rapeseed"; por isso os de "rape" são formas exatas no regex abaixo).
BLOCKED_STEMS = ["estupr", "pedofil", "pedófil", "pedophil", "paedophil", "necrofil", "zoofil", r"rap(?:e|es|ed|ing|ist|ists)(?!\w)"]
