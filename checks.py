"""Checagens automáticas de formato e de fidelidade ao pedido (usadas pelo evaluate.py).

Para cada idioma e modo há dois pedidos:
  - "visto": lugar/personagem que existe nos moldes do dataset;
  - "novo":  lugar inventado (não aparece em nenhum exemplo de treino). Mede se o modelo COPIA o nome do pedido
             em vez de recitar os lugares que decorou.
Uma amostra é "ok" se: terminou sozinha (EOS), contém os trechos exigidos, não tem linhas duplicadas
e (rumores) tem pelo menos 3 itens numerados.
"""
import re

# Trechos exigidos no modo família (em minúsculas): cabeçalho "Família X (n membros)" e os campos de cada NPC.
FAMILY_LABELS = {
    "pt": ["família ", "nome:", "ocupação:", "papel:", "personalidade:", "segredo:", "gancho da família:"],
    "en": [" family (", "name:", "occupation:", "role:", "personality:", "secret:", "family hook:"],
}

# modo -> [(pedido, trechos exigidos em minúsculas, itens numerados mínimos)]
CASES = {
    "pt": {
        "conto": [
            ("Um conto sombrio sobre Lysa, a maga, na Torre do Corvo.", ["torre do corvo", "lysa"], 0),
            ("Um conto épico sobre Ilara, a ladina, na Fortaleza de Cristal.", ["fortaleza de cristal", "ilara"], 0)],
        "cenario": [
            ("Descreva o Pântano de Vidro.", ["pântano de vidro", "gancho para o mestre"], 0),
            ("Descreva a Fortaleza de Cristal.", ["fortaleza de cristal", "gancho para o mestre"], 0)],
        # NPC: o lugar NÃO é exigido no texto. No dataset ele só aparece em algumas variantes de Segredo/Gancho
        # (~1 em cada 3 exemplos), então exigir o nome reprovaria até exemplos do próprio dataset.
        "npc": [
            ("Crie um NPC: ferreira no Porto de Âmbar.", ["nome:", "ocupação:", "aparência:", "personalidade:", "segredo:", "gancho:"], 0),
            ("Crie um NPC: mercador na Fortaleza de Cristal.", ["nome:", "ocupação:", "aparência:", "personalidade:", "segredo:", "gancho:"], 0)],
        # missão: o lugar NÃO é exigido; no dataset só aparece em parte dos objetivos/títulos, e exigir reprovaria
        # até exemplos do próprio treino (mesmo erro que tínhamos no NPC)
        "missao": [
            ("Crie uma missão para um grupo de nível 3 na Cidadela de Ferro.",
             ["título:", "objetivo:", "complicação:", "recompensa:"], 0),
            ("Crie uma missão para um grupo de nível 5 na Fortaleza de Cristal.",
             ["título:", "objetivo:", "complicação:", "recompensa:"], 0)],
        "rumor": [
            ("Quais são os rumores na taverna sobre a Ilha dos Faróis Mortos?", ["ilha dos faróis mortos"], 3),
            ("Quais são os rumores na taverna sobre a Fortaleza de Cristal?", ["fortaleza de cristal"], 3)],
        # família: (pedido, trechos, itens numerados, nº de membros exigido)
        "familia": [
            ("Crie uma família de três NPCs: ferreiros no Porto de Âmbar.", FAMILY_LABELS["pt"], 0, 3),
            ("Crie uma família de quatro NPCs: anões para a Fortaleza de Cristal.", FAMILY_LABELS["pt"], 0, 4)],
        # poema (só PT): (pedido, trechos, itens numerados, membros, mínimo de versos). "novo" = título de RPG
        "poema": [
            ('Escreva um poema chamado "Retrato".', [], 0, 0, 4),
            ('Escreva um poema chamado "A Torre do Corvo".', [], 0, 0, 4)],
    },
    "en": {
        "conto": [
            ("A grim tale about Lysa, the mage, in the Raven Tower.", ["raven tower", "lysa"], 0),
            ("An epic tale about Ilara, the rogue, in the Crystal Fortress.", ["crystal fortress", "ilara"], 0)],
        "cenario": [
            ("Describe the Glass Marsh.", ["glass marsh", "gm hook"], 0),
            ("Describe the Crystal Fortress.", ["crystal fortress", "gm hook"], 0)],
        "npc": [
            ("Create an NPC: blacksmith in the Amber Harbor.", ["name:", "occupation:", "appearance:", "personality:", "secret:", "hook:"], 0),
            ("Create an NPC: merchant in the Crystal Fortress.", ["name:", "occupation:", "appearance:", "personality:", "secret:", "hook:"], 0)],
        "missao": [
            ("Create a quest for a level 3 party in the Iron Citadel.",
             ["title:", "objective:", "complication:", "reward:"], 0),
            ("Create a quest for a level 5 party in the Crystal Fortress.",
             ["title:", "objective:", "complication:", "reward:"], 0)],
        "rumor": [
            ("What are the tavern rumors about the Isle of Dead Lighthouses?", ["isle of dead lighthouses"], 3),
            ("What are the tavern rumors about the Crystal Fortress?", ["crystal fortress"], 3)],
        "familia": [
            ("Create a family of three NPCs: blacksmiths in the Amber Harbor.", FAMILY_LABELS["en"], 0, 3),
            ("Create a family of four NPCs: dwarves for the Crystal Fortress.", FAMILY_LABELS["en"], 0, 4)],
    },
}


# Rótulos que cada modo deve ter (usado para pedidos arbitrários, ex.: geração de pares do DPO).
LABELS = {
    "pt": {"conto": [], "rumor": [], "cenario": ["gancho para o mestre"],
           "npc": ["nome:", "ocupação:", "aparência:", "personalidade:", "segredo:", "gancho:"],
           "missao": ["título:", "gancho:", "objetivo:", "complicação:", "recompensa:"],
           "familia": FAMILY_LABELS["pt"]},
    "en": {"conto": [], "rumor": [], "cenario": ["gm hook"],
           "npc": ["name:", "occupation:", "appearance:", "personality:", "secret:", "hook:"],
           "missao": ["title:", "hook:", "objective:", "complication:", "reward:"],
           "familia": FAMILY_LABELS["en"]},
}


def requirements(lang: str, mode: str, request: str, place: str | None, hero: str | None):
    """(trechos exigidos, itens numerados) para um pedido qualquer.
    Lugar/herói só são exigidos em conto e cenário (no dataset sempre aparecem) e se estiverem no pedido."""
    req = list(LABELS[lang][mode])
    if mode in ("conto", "cenario"):
        for x in (place, hero):
            if x and x.lower() in request.lower():
                req.append(x.lower())
    return req, (3 if mode == "rumor" else 0)


_NUM = {"dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4, "two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4}


def check_request(lang: str, mode: str, request: str, text: str, finished: bool) -> list[str] | None:
    """Checagem de um pedido qualquer, para o app (gerar, conferir, tentar de novo). Devolve a lista de problemas
    ([] = ok) ou None se o modo não tem checagem (poema). Família: lê a quantidade pedida ('três NPCs')."""
    if mode not in LABELS[lang]:
        return None
    members = 0
    if mode == "familia":
        m = re.search(r"\b(dois|duas|três|tres|quatro|two|three|four|[234])\b", request.lower())
        members = _NUM[m.group(1)] if m else 0
    out = problems(text, finished, list(LABELS[lang][mode]), 3 if mode == "rumor" else 0, members)
    if mode in ("npc", "familia"):
        out += bad_occupations(lang, request, text)
    return out


# Ocupações que existem no dataset de treino. Qualquer outra só vale se vier do pedido ("pedreiros" -> "pedreiro"/"pedreira"):
# com poucas ocupações vistas, o modelo inventa palavras ao imitar a que o usuário pediu ("pedeiro", "pediadora").
OCCUPATIONS = {
    "pt": {"cartógrafa", "cartógrafo", "elfa", "elfo", "mercador", "mercadora", "estalajadeira", "estalajadeiro", "guarda",
           "sacerdote", "sacerdotisa", "anão", "anã", "halfling", "ferreiro", "ferreira", "curandeira", "curandeiro",
           "contrabandista"},
    "en": {"guard", "healer", "blacksmith", "innkeeper", "cartographer", "merchant", "dwarf", "elf", "smuggler", "halfling",
           "priest"},
}


def _stem(w: str) -> str:
    """Tira plural e flexão de gênero: pedreiros, pedreira, pedreiro -> pedreir."""
    w = w.lower()
    if w.endswith("s") and len(w) > 3:
        w = w[:-1]
    if w[-1:] in "oae" and len(w) > 3:
        w = w[:-1]
    return w


def bad_occupations(lang: str, request: str, text: str) -> list[str]:
    """Aponta 'Ocupação:' com palavra que não é do dataset nem do pedido (provável palavra inventada)."""
    known = {_stem(w) for w in OCCUPATIONS[lang]} | {_stem(w) for w in re.findall(r"\w+", request)}
    out = []
    for m in re.finditer(r"^\s*(?:ocupação|occupation):\s*(.+)$", text, re.I | re.M):
        val = m.group(1).strip().rstrip(".")
        if _stem(val) not in known:
            out.append(f"ocupação desconhecida '{val}'")
    return out


def blocked_in(text: str) -> list[str]:
    """Palavras bloqueadas que aparecem no texto (palavra inteira, sem diferenciar maiúsculas)."""
    from common import BLOCKED_STEMS, BLOCKED_WORDS
    low = text.lower()
    hits = [w for w in BLOCKED_WORDS if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low)]
    # radicais: pegam qualquer flexão ("estuprada", "estupraram", "pedofilia"...) mesmo que não esteja na lista
    hits += [m.group(0) for s in BLOCKED_STEMS for m in re.finditer(r"(?<!\w)" + s + r"\w*", low)]
    return hits


def problems(text: str, finished: bool, required: list[str], numbered: int, members: int = 0,
             min_lines: int = 0) -> list[str]:
    """members > 0 (modo família): exige exatamente esse número de NPCs, todos com o mesmo sobrenome e nomes diferentes.
    min_lines > 0 (modo poema): exige esse mínimo de versos; versos repetidos (refrão) são permitidos."""
    low, out = text.lower(), []
    if blocked_in(text):
        out.append("palavra bloqueada")
    if not finished:
        out.append("não terminou")
    out += [f"falta '{x}'" for x in required if x not in low]
    lines = [re.sub(r"^\s*\d+[.\-–]+\s*", "", l).strip() for l in text.splitlines() if l.strip()]
    if members:
        # na família linhas iguais são legítimas (dois "Ocupação: ferreiro"); o que não pode é nome repetido
        names = [l.split(":", 1)[1].strip() for l in text.splitlines() if re.match(r"^\s*(nome|name):", l, re.I)]
        head = re.search(r"\((\d)\s+(?:membros|members)\)\s*:\s*(.+)", text)
        roster = [x.strip() for x in head.group(2).split(",") if x.strip()] if head else []
        if len(roster) != members:
            out.append(f"lista de nomes no cabeçalho com {len(roster)} em vez de {members}")
        if len(names) != members:
            out.append(f"{len(names)} NPCs em vez de {members}")
        else:
            if len({n.split()[-1] for n in names if n.split()}) != 1:
                out.append("sobrenomes diferentes")
            if len(set(names)) != len(names):
                out.append("nomes repetidos")
    elif min_lines:
        if len(lines) < min_lines:
            out.append(f"só {len(lines)} versos")
        if "###" in text:
            out.append("vazou o formato")
    elif len(lines) != len(set(lines)):
        out.append("linha duplicada")
    if numbered:
        n = len([l for l in text.splitlines() if re.match(r"^\s*\d+[.\-–]", l)])
        if n < numbered:
            out.append(f"só {n} itens")
    return out



def repair_occupations(lang: str, request: str, text: str) -> tuple[str, int]:
    """Troca a ocupação inventada ('extirpe', 'exíva') pela palavra do pedido que ela tentava imitar
    ('extirpador'), na flexão de gênero que o modelo usou. Devolve (texto, nº de trocas).
    Só mexe em ocupação desconhecida (fora do dataset e do pedido) e só se alguma palavra do pedido for parecida."""
    import difflib
    cands = []
    for w in re.findall(r"\w+", request):
        w = w.lower()
        if len(w) < 5:
            continue
        if w.endswith(("ores", "ares", "ões")):
            w = w[:-2] if w.endswith("es") else w[:-3] + "ão"
        elif w.endswith("s"):
            w = w[:-1]
        cands.append(w)
    known = {_stem(w) for w in OCCUPATIONS[lang]} | {_stem(w) for w in re.findall(r"\w+", request)}
    n = 0

    def fix(m):
        nonlocal n
        val = m.group(2).strip().rstrip(".")
        if _stem(val) in known or not cands:
            return m.group(0)
        best = max(cands, key=lambda c: difflib.SequenceMatcher(None, val.lower(), c).ratio())
        if difflib.SequenceMatcher(None, val.lower(), best).ratio() < 0.5:
            return m.group(0)
        fem = val.lower().endswith("a") and lang == "pt"
        if fem and best.endswith("o"):
            best = best[:-1] + "a"
        elif fem and best.endswith("or"):
            best += "a"
        n += 1
        return m.group(1) + best

    return re.sub(r"^(\s*(?:ocupação|occupation):\s*)(.+)$", fix, text, flags=re.I | re.M), n
