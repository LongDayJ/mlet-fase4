"""Gera o dataset sintético bilíngue (PT/EN) de fantasia para RPG.

Uso:
    python data/build_dataset.py                 # gera data/train.jsonl e data/val.jsonl
    python data/build_dataset.py --scale 2       # dobra a quantidade
    python data/build_dataset.py --extra data/extra   # inclui .txt reais (pt_*.txt / en_*.txt) como modo "continuar"

Cada linha: {"lang","mode","request","prompt","completion"}.
Os textos são montados combinando pools de elementos (heróis, lugares, ameaças, relíquias,
motivações, reviravoltas...) em vários templates, o que dá variedade estrutural suficiente
para ensinar formato, idioma e estilo ao modelo.
"""
import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import build_prompt  # noqa: E402


def fix(s: str) -> str:
    """Capitaliza início do texto e após . ! ?"""
    return re.sub(r"(^|[.!?]\s+)([a-zà-ú])", lambda m: m.group(1) + m.group(2).upper(), s)


def np_pt(g, n):
    a, i, t, o = ("a", "na", "à", "da") if g == "f" else ("o", "no", "ao", "do")
    return {"the": f"{a} {n}", "in": f"{i} {n}", "to": f"{t} {n}", "of": f"{o} {n}"}


def np_en(g, n):
    return {"the": f"the {n}", "in": f"in the {n}", "to": f"to the {n}", "of": f"of the {n}"}


NAMES = [("Aldric", "m"), ("Lysa", "f"), ("Doran", "m"), ("Isolde", "f"), ("Eryndor", "m"), ("Selene", "f"),
         ("Garrick", "m"), ("Nyla", "f"), ("Tarek", "m"), ("Vesper", "f"), ("Orin", "m"), ("Brisa", "f"),
         ("Caius", "m"), ("Ysolde", "f"), ("Fenrick", "m"), ("Livia", "f")]

PT = {
    "np": np_pt,
    "names": NAMES,
    "classes": [("ladino", "ladina"), ("paladino", "paladina"), ("mago", "maga"), ("bardo", "barda"),
                ("arqueiro", "arqueira"), ("clérigo", "clériga"), ("guerreiro", "guerreira"),
                ("druida", "druida"), ("caçador", "caçadora"), ("alquimista", "alquimista")],
    "cls_the": lambda c, g: ("a " + c[1]) if g == "f" else ("o " + c[0]),
    "places": [("f", "Torre do Corvo"), ("f", "Floresta de Névoa Cinzenta"), ("m", "Vale das Sombras"),
               ("f", "Cidadela de Ferro"), ("m", "Pântano de Vidro"), ("f", "Cripta do Rei Esquecido"),
               ("m", "Porto de Âmbar"), ("f", "Montanha do Dragão Adormecido"), ("m", "Bosque dos Sussurros"),
               ("f", "Ruína de Aurelath"), ("m", "Mercado dos Mil Olhos"), ("f", "Taverna do Javali Torto"),
               ("m", "Monastério do Silêncio"), ("f", "Ilha dos Faróis Mortos"), ("m", "Deserto de Cinzas"),
               ("f", "Biblioteca Submersa")],
    "threats": [("f", "bruxa dos pântanos"), ("m", "dragão de escamas negras"), ("m", "necromante exilado"),
                ("f", "rainha-aranha"), ("m", "cavaleiro sem rosto"), ("f", "horda de goblins"),
                ("m", "lich adormecido"), ("f", "serpente marinha"), ("m", "culto da Lua Pálida"),
                ("f", "sombra devoradora")],
    "relics": [("f", "Coroa de Espinhos Prateados"), ("f", "Espada Lamento"), ("m", "Grimório de Cinzas"),
               ("m", "Amuleto do Crepúsculo"), ("f", "Chave de Osso"), ("m", "Cálice das Marés"),
               ("f", "Lanterna de Almas"), ("m", "Mapa das Estrelas Cadentes")],
    "motives": ["salvar a aldeia onde nasceu", "pagar uma dívida antiga", "encontrar o irmão desaparecido",
                "recuperar a honra perdida", "quebrar uma maldição de família",
                "descobrir quem traiu a sua ordem", "proteger um segredo que não lhe pertencia"],
    "omens": ["cheiro de ferro e cinzas", "um frio que não pertencia àquela estação",
              "o eco distante de sinos que ninguém tocava", "o perfume de flores que já deveriam ter morrido",
              "um murmúrio em uma língua esquecida"],
    "comps": ["um corvo falante", "uma gata de três olhos", "um anão rabugento",
              "uma jovem ladra de sorriso fácil", "um golem de pedra que só sabia cantar",
              "um velho cavaleiro enferrujado", "uma raposa que entendia todas as línguas"],
    "tones": [("sombrio", "sombria", "Nada naquela noite foi gentil, e todos sabiam disso."),
              ("épico", "épica", "Era o tipo de feito que os bardos cantariam por mil anos."),
              ("leve", "leve", "Tudo isso, claro, antes do café da manhã."),
              ("misterioso", "misteriosa", "E havia algo mais, algo que ninguém ousou contar.")],
    "req_conto": ["Um conto {tm} sobre {h}, {cls}, {pl[in]}.",
                  "Escreva uma história {tf} em que {h} precisa {motive}.",
                  "Conto de fantasia: {th[the]} e {re[the]}."],
    "p1": ["{pl[in]}, {h}, {cls}, chegou ao anoitecer com um único propósito: {motive}. "
           "O vento trazia {omen}, e ninguém por ali ousava falar em voz alta.",
           "Ninguém lembrava quando {th[the]} se instalou {pl[in]}, mas {h}, {cls}, tinha um objetivo claro: {motive}.",
           "A estrada até {pl[the]} era longa, e {h}, {cls}, já tinha perdido a conta das noites mal dormidas. "
           "Ainda assim, precisava {motive}."],
    "p2": ["Ao lado de {comp}, {h} descobriu que {th[the]} guardava {re[the]} e que muitos já haviam "
           "desaparecido tentando se aproximar.",
           "A jornada foi dura. Ao lado de {comp}, {h} escapou de duas emboscadas antes de enfrentar {th[the]}, "
           "que protegia {re[the]} havia séculos.",
           "No coração do lugar, {th[the]} esperava em silêncio, vigiando {re[the]}. {comp} sussurrou que ainda "
           "dava tempo de voltar, mas {h} já tinha ido longe demais."],
    "twists": ["Mas a verdade era outra: {comp} servia {th[the]} desde o início, e a viagem inteira fora uma armadilha.",
               "Quando {h} finalmente tocou {re[the]}, percebeu que aquilo não era um prêmio, e sim uma prisão, "
               "e que {th[the]} era o seu último carcereiro.",
               "Ao encarar {th[the]}, {h} compreendeu que o verdadeiro inimigo era o medo de falhar, "
               "e que ele morava no seu peito desde o começo.",
               "Então o chão tremeu, e {h} percebeu que todo aquele lugar estava vivo e observava cada passo."],
    "choices": ["a misericórdia em vez da vingança", "o sacrifício em vez da glória", "a verdade em vez do conforto"],
    "endings": ["No fim, {h} escolheu {choice}. Quando o sol nasceu, nada {pl[in]} voltou a ser como antes, "
                "e as canções da região passaram a levar o seu nome.",
                "{h} partiu ao amanhecer sem olhar para trás e, desde então, dizem que {re[the]} ainda sussurra o seu nome.",
                "A aventura terminou, mas {comp} jurou que aquilo era só o começo."],
    "short_endings": ["{h} partiu ao amanhecer sem olhar para trás, e as canções da região passaram a levar o seu nome.",
                      "No fim, {h} escolheu {choice}, e nada {pl[in]} voltou a ser como antes.",
                      "Quando o sol nasceu, {h} já estava longe, mas ninguém {pl[in]} esqueceu o que aconteceu ali."],
    # cenário
    "sims": ["uma ferida aberta na paisagem", "um sonho que alguém esqueceu de terminar",
             "um velho animal adormecido", "uma cidade que se recusa a morrer"],
    "smells": ["musgo e ferro velho", "fumaça de lenha úmida", "sal e peixe podre", "incenso queimado há muito tempo"],
    "sounds": ["ao longe, alguém repete a mesma canção sem parar",
               "a água goteja em um ritmo que parece uma contagem regressiva",
               "nenhum pássaro canta", "algo grande se move debaixo do chão"],
    "gm_hooks": ["um teste de Percepção revela pegadas recentes levando para dentro",
                 "quem passa a noite aqui sonha com o mesmo rosto",
                 "um mapa antigo escondido sob uma pedra solta aponta para {re[the]}",
                 "{comp} oferece guiar o grupo, mas cobra um preço estranho"],
    "req_cen": ["Descreva {pl[the]}.", "Como é {pl[the]}?"],
    "cen": ["{pl[the]} se estende diante dos viajantes como {sim}.",
            "O ar cheira a {smell}, e {sound}.",
            "Os moradores dizem que {th[the]} vive ali e que {re[the]} ainda repousa em algum canto, "
            "esperando por quem for corajoso ou tolo o bastante."],
    "cen_gm": "Gancho para o mestre",
    # npc
    "roles": [("ferreiro", "ferreira"), ("estalajadeiro", "estalajadeira"), ("mercador", "mercadora"),
              ("guarda", "guarda"), ("curandeiro", "curandeira"), ("cartógrafo", "cartógrafa"),
              ("contrabandista", "contrabandista"), ("sacerdote", "sacerdotisa")],
    "looks": ["uma cicatriz que cruza a sobrancelha esquerda", "olhos de cores diferentes",
              "mãos sempre sujas de tinta", "um manto remendado em dez tons de azul"],
    "traits": ["desconfia de todos, mas nunca de crianças", "conta piadas ruins quando está nervoso",
               "fala sempre baixo, como se alguém ouvisse", "é generoso até demais"],
    "secrets": ["deve um favor perigoso {th[to]}", "guarda um mapa que leva até {re[the]}",
                "descende da família que construiu {pl[the]}",
                "viu {th[the]} de perto e jurou nunca contar a ninguém"],
    "npc_hooks": ["Vende informações sobre {th[the]} em troca de um favor.",
                  "Oferece abrigo ao grupo, mas pede discrição.",
                  "Precisa urgentemente de alguém para entregar uma carta {pl[to]}."],
    "req_npc": ["Crie um NPC: {role} {pl[in]}.", "Preciso de um NPC {pl[in]}: {role}."],
    "npc_labels": ["Nome", "Ocupação", "Aparência", "Personalidade", "Segredo", "Gancho"],
    # missão
    "titles": ["O Segredo {re[of]}", "A Ameaça {th[of]}", "Sombras {pl[in]}"],
    "goals": ["recuperar {re[the]} antes do próximo eclipse", "impedir {th[the]} de chegar {pl[to]}",
              "descobrir o que aconteceu {pl[in]} e voltar vivo"],
    "quest_hook": "Um mensageiro ferido chega à cidade com um pedido desesperado: {goal}.",
    "complications": ["{th[the]} já sabe que o grupo está a caminho",
                      "uma ponte desabou e o único caminho passa por território hostil",
                      "um dos contratantes é, na verdade, um espião",
                      "o tempo está contando: ao amanhecer do terceiro dia será tarde demais"],
    "rewards": ["{n} moedas de ouro e a gratidão de um nobre local",
                "um item mágico menor e acesso à biblioteca do templo",
                "{n} moedas de ouro e um favor que pode ser cobrado a qualquer momento"],
    "req_quest": ["Crie uma missão para um grupo de nível {lvl} {pl[in]}."],
    "quest_labels": ["Título", "Gancho", "Objetivo", "Complicação", "Recompensa"],
    # rumores
    "rumors": ["Dizem que {th[the]} voltou a rondar {pl[the]}.",
               "Um mercador jura ter visto {re[the]} à venda por uma moeda de cobre.",
               "Alguém ouviu cantos vindos {pl[of]} em plena madrugada.",
               "Contam que quem dorme {pl[in]} acorda com um nome que não é o seu.",
               "Os guardas pagam bem por qualquer informação sobre {comp}.",
               "O velho taverneiro afirma que {th[the]} tem medo {re[of]}."],
    "req_rumor": ["Quais são os rumores na taverna sobre {pl[the]}?"],
    "extra_req": "Continue a história: {start}",
}

EN = {
    "np": np_en,
    "names": NAMES,
    "classes": [(c, c) for c in ["rogue", "paladin", "mage", "bard", "archer", "cleric", "warrior", "druid",
                                 "hunter", "alchemist"]],
    "cls_the": lambda c, g: "the " + c[0],
    "places": [("f", "Raven Tower"), ("f", "Grey Mist Forest"), ("m", "Shadow Vale"), ("f", "Iron Citadel"),
               ("m", "Glass Marsh"), ("f", "Crypt of the Forgotten King"), ("m", "Amber Harbor"),
               ("f", "Mountain of the Sleeping Dragon"), ("m", "Whispering Grove"), ("f", "Ruins of Aurelath"),
               ("m", "Market of a Thousand Eyes"), ("f", "Crooked Boar Tavern"), ("m", "Monastery of Silence"),
               ("f", "Isle of Dead Lighthouses"), ("m", "Ash Desert"), ("f", "Sunken Library")],
    "threats": [("f", "marsh witch"), ("m", "black-scaled dragon"), ("m", "exiled necromancer"),
                ("f", "spider queen"), ("m", "faceless knight"), ("f", "goblin horde"), ("m", "sleeping lich"),
                ("f", "sea serpent"), ("m", "cult of the Pale Moon"), ("f", "devouring shadow")],
    "relics": [("f", "Crown of Silver Thorns"), ("f", "Blade of Lament"), ("m", "Grimoire of Ashes"),
               ("m", "Amulet of Dusk"), ("f", "Bone Key"), ("m", "Chalice of Tides"),
               ("f", "Lantern of Souls"), ("m", "Map of Falling Stars")],
    "motives": ["save the village where they grew up", "repay an old debt", "find a missing brother",
                "win back a lost honor", "break a family curse", "discover who betrayed their order",
                "guard a secret that was never theirs"],
    "omens": ["the smell of iron and ash", "a cold that did not belong to the season",
              "the distant echo of bells no one was ringing", "the scent of flowers that should have died long ago",
              "a murmur in a forgotten tongue"],
    "comps": ["a talking raven", "a three-eyed cat", "a grumpy dwarf", "a young thief with an easy smile",
              "a stone golem who only knew how to sing", "an old rusty knight", "a fox that understood every language"],
    "tones": [("grim", "grim", "Nothing about that night was kind, and everyone knew it."),
              ("epic", "epic", "It was the kind of deed bards would sing about for a thousand years."),
              ("lighthearted", "lighthearted", "All of this, of course, before breakfast."),
              ("mysterious", "mysterious", "And there was something more, something no one dared to tell.")],
    "req_conto": ["{A} {tm} tale about {h}, {cls}, {pl[in]}.",
                  "Write {a} {tf} story in which {h} must {motive}.",
                  "Fantasy tale: {th[the]} and {re[the]}."],
    "p1": ["{pl[in]}, {h}, {cls}, arrived at nightfall with a single purpose: to {motive}. "
           "The wind carried {omen}, and no one dared to speak above a whisper.",
           "No one remembered when {th[the]} settled {pl[in]}, but {h}, {cls}, had a clear goal: to {motive}.",
           "The road to {pl[the]} was long, and {h}, {cls}, had lost count of the sleepless nights. "
           "Still, {h} had to {motive}."],
    "p2": ["Alongside {comp}, {h} learned that {th[the]} guarded {re[the]}, and that many had vanished "
           "trying to get close.",
           "The journey was hard. Alongside {comp}, {h} escaped two ambushes before facing {th[the]}, "
           "which had protected {re[the]} for centuries.",
           "In the heart of the place, {th[the]} waited in silence, watching over {re[the]}. {comp} whispered "
           "that it was not too late to turn back, but {h} had already come too far."],
    "twists": ["But the truth was different: {comp} had served {th[the]} from the start, and the whole journey "
               "had been a trap.",
               "When {h} finally touched {re[the]}, it became clear that it was no prize but a prison, "
               "and that {th[the]} was its last jailer.",
               "Facing {th[the]}, {h} understood that the true enemy was the fear of failing, "
               "and that it had been living in their own chest all along.",
               "Then the ground trembled, and {h} realized that the whole place was alive and had been "
               "watching every step."],
    "choices": ["mercy over revenge", "sacrifice over glory", "truth over comfort"],
    "endings": ["In the end, {h} chose {choice}. When the sun rose, {pl[the]} was never the same, "
                "and the songs of the region began to carry {h}'s name.",
                "{h} left at dawn without looking back, and ever since, they say {re[the]} still whispers {h}'s name.",
                "The adventure was over, but {comp} swore it was only the beginning."],
    "short_endings": ["{h} left at dawn without looking back, and the songs of the region began to carry {h}'s name.",
                      "In the end, {h} chose {choice}, and {pl[the]} was never the same.",
                      "By the time the sun rose, {h} was long gone, but no one in {pl[the]} forgot what happened there."],
    "sims": ["an open wound in the landscape", "a dream someone forgot to finish", "an old sleeping beast",
             "a city that refuses to die"],
    "smells": ["moss and old iron", "damp wood smoke", "salt and rotten fish", "incense burned long ago"],
    "sounds": ["far away, someone keeps repeating the same song", "water drips in a rhythm like a countdown",
               "not a single bird sings", "something large shifts beneath the ground"],
    "gm_hooks": ["a Perception check reveals fresh footprints leading inside",
                 "anyone who sleeps here dreams of the same face",
                 "an old map hidden under a loose stone points to {re[the]}",
                 "{comp} offers to guide the party, but asks for a strange price"],
    "req_cen": ["Describe {pl[the]}.", "What is {pl[the]} like?"],
    "cen": ["{pl[the]} stretches out before the travellers like {sim}.",
            "The air smells of {smell}, and {sound}.",
            "Locals say {th[the]} lives there and that {re[the]} still rests in some corner, "
            "waiting for someone brave or foolish enough."],
    "cen_gm": "GM hook",
    "roles": [(r, r) for r in ["blacksmith", "innkeeper", "merchant", "guard", "healer", "cartographer",
                               "smuggler", "priest"]],
    "looks": ["a scar crossing the left eyebrow", "eyes of different colors", "hands always stained with ink",
              "a cloak patched in ten shades of blue"],
    "traits": ["distrusts everyone, but never children", "tells bad jokes when nervous",
               "always speaks softly, as if someone were listening", "is generous to a fault"],
    "secrets": ["owes a dangerous favor to {th[the]}", "keeps a map that leads to {re[the]}",
                "descends from the family that built {pl[the]}",
                "saw {th[the]} up close and swore never to tell anyone"],
    "npc_hooks": ["Sells information about {th[the]} in exchange for a favor.",
                  "Offers the party shelter, but asks for discretion.",
                  "Urgently needs someone to deliver a letter {pl[to]}."],
    "req_npc": ["Create an NPC: {role} {pl[in]}.", "I need an NPC {pl[in]}: {role}."],
    "npc_labels": ["Name", "Occupation", "Appearance", "Personality", "Secret", "Hook"],
    "titles": ["The Secret {re[of]}", "The Threat {th[of]}", "Shadows {pl[in]}"],
    "goals": ["recover {re[the]} before the next eclipse", "stop {th[the]} from reaching {pl[the]}",
              "find out what happened {pl[in]} and return alive"],
    "quest_hook": "A wounded messenger arrives in town with a desperate plea: {goal}.",
    "complications": ["{th[the]} already knows the party is on the way",
                      "a bridge has collapsed and the only path crosses hostile territory",
                      "one of the clients is actually a spy",
                      "time is running out: by dawn on the third day it will be too late"],
    "rewards": ["{n} gold coins and the gratitude of a local noble",
                "a minor magic item and access to the temple library",
                "{n} gold coins and a favor that can be called in at any time"],
    "req_quest": ["Create a quest for a level {lvl} party {pl[in]}."],
    "quest_labels": ["Title", "Hook", "Objective", "Complication", "Reward"],
    "rumors": ["They say {th[the]} has returned to haunt {pl[the]}.",
               "A merchant swears he saw {re[the]} on sale for a single copper coin.",
               "Someone heard singing coming from {pl[the]} in the dead of night.",
               "They say whoever sleeps {pl[in]} wakes up with a name that is not theirs.",
               "The guards pay well for any information about {comp}.",
               "The old tavern keeper claims {th[the]} is afraid of {re[the]}."],
    "req_rumor": ["What are the tavern rumors about {pl[the]}?"],
    "extra_req": "Continue the story: {start}",
}

PACKS = {"pt": PT, "en": EN}

# Variedade extra, somada aos pools acima. Regras: p1/twists/short_endings não citam {comp} nem {re}
# (a versão curta não apresenta o companheiro); em PT usar {x[in]}/{x[of]}/{x[to]} para evitar "em a", "de a".
EXTRA = {
    "pt": {
        "names": [("Ilena", "f"), ("Thalion", "m"), ("Kaelen", "m"), ("Soren", "m"), ("Elara", "f"),
                  ("Bruna", "f"), ("Dagan", "m"), ("Anwen", "f")],
        "places": [("f", "Ponte dos Enforcados"), ("m", "Castelo de Cinzas"), ("f", "Aldeia dos Espelhos"),
                   ("m", "Labirinto de Raízes"), ("f", "Gruta dos Ecos"), ("m", "Templo do Sol Partido")],
        "threats": [("m", "gigante de gelo"), ("f", "hidra de três cabeças"), ("m", "fantasma do Barão Vermelho"),
                    ("f", "matilha de lobos espectrais")],
        "relics": [("f", "Máscara de Cera"), ("m", "Anel da Meia-Noite"), ("f", "Harpa de Vento"),
                   ("m", "Escudo de Bronze Antigo")],
        "motives": ["vingar a morte do mentor", "encontrar um lugar para chamar de lar",
                    "entregar uma carta antes do solstício"],
        "omens": ["o gosto metálico de uma tempestade distante", "o rastro de uma música que ninguém reconhecia",
                  "a luz pálida de uma lua que ainda não tinha nascido"],
        "comps": ["um menino que nunca falava", "uma mula teimosa chamada Pimenta", "um espectro educado demais"],
        "p1": ["Ao entrar {pl[in]}, {h}, {cls}, notou {omen}. Era preciso {motive}.",
               "Dizem que {th[the]} ainda vigia {pl[the]}. {h}, {cls}, não acreditava em histórias, "
               "mas precisava {motive}."],
        "p2": ["{comp} conhecia um atalho, mas {th[the]} já esperava no fim dele, de olho {re[in]}. "
               "{h} respirou fundo e seguiu em frente.",
               "Foi {comp} quem notou primeiro que {re[the]} brilhava sob a vigia {th[of]}. "
               "{h} sabia que não haveria segunda chance."],
        "twists": ["Mas, ao chegar perto, {h} percebeu que {th[the]} chorava: tudo o que queria era ser libertado.",
                   "{h} então entendeu que {th[the]} jamais fora o verdadeiro inimigo, e sim a última "
                   "testemunha de um segredo antigo.",
                   "De repente, o silêncio {pl[of]} se partiu, e {h} ouviu o próprio nome sussurrado por todos os lados."],
        "endings": ["{h} guardou o segredo para si. Anos depois, viajantes ainda juravam ouvir passos {pl[in]} "
                    "nas noites sem lua.",
                    "Ninguém soube o que {h} deixou para trás, mas {comp} afirmava ter visto uma luz acesa "
                    "{pl[in]} até o amanhecer."],
        "short_endings": ["{h} nunca mais foi visto, mas dizem que {pl[the]} guarda até hoje a sua sombra."],
        "sims": ["um corredor sem fim feito de pedra e silêncio", "uma festa que terminou há séculos sem avisar os convidados"],
        "smells": ["chuva antiga e ervas amargas", "cera derretida e poeira"],
        "sounds": ["sinos distantes tocam fora de hora", "passos leves seguem o grupo e param quando o grupo para"],
        "gm_hooks": ["um teste de Investigação revela uma porta escondida atrás de um tapete apodrecido",
                     "uma criança perdida pede ajuda, mas ninguém na região a conhece"],
        "looks": ["uma voz rouca que parece vir de muito longe", "dedos tortos de tanto trabalho",
                  "um sorriso largo com um dente de ouro"],
        "traits": ["nunca esquece um rosto, mas finge esquecer", "reza em voz alta antes de cada negócio",
                   "coleciona objetos quebrados"],
        "secrets": ["foi condenado em outra cidade e mudou de nome", "escreve cartas para alguém que já morreu"],
        "npc_hooks": ["Conhece uma passagem secreta que ninguém mais conhece.",
                      "Pede ao grupo que encontre um objeto perdido."],
        "complications": ["um dos aliados do grupo some na primeira noite",
                          "o clima piora e todos os caminhos ficam intransitáveis"],
        "rewards": ["um mapa raro e um cavalo de viagem", "{n} moedas de ouro e uma dívida perdoada"],
        "goals": ["escoltar um mensageiro até {pl[the]} sem ser visto"],
        "titles": ["O Último Aviso {pl[of]}", "O Preço {re[of]}"],
        "rumors": ["Um bêbado jura que {comp} conhece o caminho secreto até {pl[the]}.",
                   "Dizem que {re[the]} muda de dono a cada lua cheia.",
                   "Contam que {th[the]} foi visto comprando suprimentos na vila."],
    },
    "en": {
        "names": [("Ilena", "f"), ("Thalion", "m"), ("Kaelen", "m"), ("Soren", "m"), ("Elara", "f"),
                  ("Bruna", "f"), ("Dagan", "m"), ("Anwen", "f")],
        "places": [("f", "Bridge of the Hanged"), ("m", "Castle of Ashes"), ("f", "Village of Mirrors"),
                   ("m", "Labyrinth of Roots"), ("f", "Cave of Echoes"), ("m", "Temple of the Broken Sun")],
        "threats": [("m", "frost giant"), ("f", "three-headed hydra"), ("m", "ghost of the Red Baron"),
                    ("f", "pack of spectral wolves")],
        "relics": [("f", "Wax Mask"), ("m", "Midnight Ring"), ("f", "Wind Harp"), ("m", "Ancient Bronze Shield")],
        "motives": ["avenge the death of a mentor", "find a place to call home",
                    "deliver a letter before the solstice"],
        "omens": ["the metallic taste of a distant storm", "the trail of a song no one recognized",
                  "the pale light of a moon that had not yet risen"],
        "comps": ["a boy who never spoke", "a stubborn mule named Pepper", "an overly polite ghost"],
        "p1": ["Entering {pl[the]}, {h}, {cls}, noticed {omen}. There was only one thing to do: {motive}.",
               "They say {th[the]} still watches over {pl[the]}. {h}, {cls}, did not believe in stories, "
               "but had to {motive}."],
        "p2": ["{comp} knew a shortcut, but {th[the]} was already waiting at the end of it, eyeing {re[the]}. "
               "{h} took a deep breath and kept going.",
               "It was {comp} who first noticed that {re[the]} glowed under the watch of {th[the]}. "
               "{h} knew there would be no second chance."],
        "twists": ["But as {h} drew closer, {th[the]} was weeping: all it wanted was to be set free.",
                   "Then {h} understood that {th[the]} had never been the true enemy, only the last witness "
                   "of an ancient secret.",
                   "Suddenly the silence {pl[of]} broke, and {h} heard their own name whispered from every side."],
        "endings": ["{h} kept the secret. Years later, travellers still swore they heard footsteps {pl[in]} "
                    "on moonless nights.",
                    "No one knew what {h} left behind, but {comp} claimed to have seen a light burning "
                    "{pl[in]} until dawn."],
        "short_endings": ["{h} was never seen again, but they say {pl[the]} still keeps {h}'s shadow."],
        "sims": ["an endless corridor of stone and silence", "a party that ended centuries ago without telling the guests"],
        "smells": ["old rain and bitter herbs", "melted wax and dust"],
        "sounds": ["distant bells ring at the wrong hour", "light footsteps follow the party and stop whenever it stops"],
        "gm_hooks": ["an Investigation check reveals a hidden door behind a rotting tapestry",
                     "a lost child asks for help, but no one in the region knows them"],
        "looks": ["a hoarse voice that seems to come from far away", "fingers bent from years of work",
                  "a wide smile with a gold tooth"],
        "traits": ["never forgets a face, but pretends to", "prays aloud before every deal",
                   "collects broken objects"],
        "secrets": ["was convicted in another city and changed names", "writes letters to someone who has died"],
        "npc_hooks": ["Knows a secret passage no one else knows.", "Asks the party to find a lost object."],
        "complications": ["one of the party's allies vanishes on the first night",
                          "the weather turns and every road becomes impassable"],
        "rewards": ["a rare map and a travelling horse", "{n} gold coins and a forgiven debt"],
        "goals": ["escort a messenger to {pl[the]} unseen"],
        "titles": ["The Last Warning {pl[of]}", "The Price {re[of]}"],
        "rumors": ["A drunk swears {comp} knows the secret path to {pl[the]}.",
                   "They say {re[the]} changes owners with every full moon.",
                   "People claim {th[the]} was seen buying supplies in the village."],
    },
}
for _lang, _extra in EXTRA.items():
    for _key, _vals in _extra.items():
        PACKS[_lang][_key] = PACKS[_lang][_key] + _vals

# Modo "familia": 2 a 4 NPCs aparentados (mesmo sobrenome, papéis na família), pedido com a quantidade.
FAMILY = {
    "pt": {
        "fam_kinds": [("anões", "anão", "anã"), ("elfos", "elfo", "elfa"), ("halflings", "halfling", "halfling"),
                      ("ferreiros", "ferreiro", "ferreira"), ("mercadores", "mercador", "mercadora"),
                      ("estalajadeiros", "estalajadeiro", "estalajadeira"), ("curandeiros", "curandeiro", "curandeira"),
                      ("guardas", "guarda", "guarda"), ("contrabandistas", "contrabandista", "contrabandista"),
                      ("cartógrafos", "cartógrafo", "cartógrafa"), ("sacerdotes", "sacerdote", "sacerdotisa")],
        "fam_rel": {"pai": "m", "mãe": "f", "filho": "m", "filha": "f", "avô": "m", "avó": "f", "irmão": "m",
                    "irmã": "f", "tio": "m", "tia": "f", "neto": "m", "neta": "f", "sobrinho": "m", "sobrinha": "f"},
        "fam_structs": {
            2: [("pai", "filha"), ("mãe", "filho"), ("avô", "neta"), ("avó", "neto"), ("irmão", "irmã"),
                ("tio", "sobrinha"), ("tia", "sobrinho")],
            3: [("pai", "mãe", "filha"), ("mãe", "filho", "filha"), ("avó", "mãe", "neto"), ("pai", "filho", "filho"),
                ("irmão", "irmã", "irmão"), ("tio", "tia", "sobrinho")],
            4: [("pai", "mãe", "filho", "filha"), ("avô", "pai", "filho", "neta"), ("mãe", "filho", "filha", "filha"),
                ("tio", "tia", "sobrinho", "sobrinha")]},
        "surnames": ["Valdris", "Korrin", "Thornwick", "Ashvale", "Brandemar", "Stonehelm", "Ravenscar", "Mooncrest",
                     "Duskmantle", "Ironvale", "Greywater", "Fenhollow"],
        "num_words": {2: "dois", 3: "três", 4: "quatro"},
        "fam_req": ["Crie uma família de {n} NPCs: {plural} {pl[in]}.",
                    "Crie uma família de {n} NPCs: {plural} para {pl[the]}.",
                    "Preciso de {n} NPCs da mesma família: {plural} {pl[in]}."],
        "fam_header": "Família {s} ({n} membros): {roster}",
        "fam_labels": ["Nome", "Ocupação", "Papel", "Personalidade", "Segredo"],
        "fam_hook_label": "Gancho da família",
        "fam_hooks": ["A família inteira deve um favor {th[to]}.",
                      "Todos guardam em segredo um mapa que leva até {re[the]}.",
                      "Uma briga antiga entre eles pode custar caro ao grupo de aventureiros.",
                      "A família pede ajuda para recuperar {re[the]} antes que {th[the]} chegue primeiro.",
                      "Escondem {comp} no porão e imploram que ninguém conte.",
                      "O chefe da família quer contratar o grupo para vigiar {pl[the]} durante a noite."],
    },
    "en": {
        "fam_kinds": [("dwarves", "dwarf", "dwarf"), ("elves", "elf", "elf"), ("halflings", "halfling", "halfling"),
                      ("blacksmiths", "blacksmith", "blacksmith"), ("merchants", "merchant", "merchant"),
                      ("innkeepers", "innkeeper", "innkeeper"), ("healers", "healer", "healer"),
                      ("guards", "guard", "guard"), ("smugglers", "smuggler", "smuggler"),
                      ("cartographers", "cartographer", "cartographer"), ("priests", "priest", "priest")],
        "fam_rel": {"father": "m", "mother": "f", "son": "m", "daughter": "f", "grandfather": "m",
                    "grandmother": "f", "brother": "m", "sister": "f", "uncle": "m", "aunt": "f",
                    "grandson": "m", "granddaughter": "f", "nephew": "m", "niece": "f"},
        "fam_structs": {
            2: [("father", "daughter"), ("mother", "son"), ("grandfather", "granddaughter"),
                ("grandmother", "grandson"), ("brother", "sister"), ("uncle", "niece"), ("aunt", "nephew")],
            3: [("father", "mother", "daughter"), ("mother", "son", "daughter"), ("grandmother", "mother", "grandson"),
                ("father", "son", "son"), ("brother", "sister", "brother"), ("uncle", "aunt", "nephew")],
            4: [("father", "mother", "son", "daughter"), ("grandfather", "father", "son", "granddaughter"),
                ("mother", "son", "daughter", "daughter"), ("uncle", "aunt", "nephew", "niece")]},
        "surnames": ["Valdris", "Korrin", "Thornwick", "Ashvale", "Brandemar", "Stonehelm", "Ravenscar", "Mooncrest",
                     "Duskmantle", "Ironvale", "Greywater", "Fenhollow"],
        "num_words": {2: "two", 3: "three", 4: "four"},
        "fam_req": ["Create a family of {n} NPCs: {plural} in {pl[the]}.",
                    "Create a family of {n} NPCs: {plural} for {pl[the]}.",
                    "I need {n} NPCs from the same family: {plural} in {pl[the]}."],
        "fam_header": "The {s} family ({n} members): {roster}",
        "fam_labels": ["Name", "Occupation", "Role", "Personality", "Secret"],
        "fam_hook_label": "Family hook",
        "fam_hooks": ["The whole family owes a favor to {th[the]}.",
                      "They all keep a secret map that leads to {re[the]}.",
                      "An old feud between them could cost the party dearly.",
                      "The family asks for help recovering {re[the]} before {th[the]} gets there first.",
                      "They hide {comp} in the cellar and beg everyone not to tell.",
                      "The family head wants to hire the party to watch over {pl[the]} through the night."],
    },
}
for _lang, _fam in FAMILY.items():
    PACKS[_lang].update(_fam)

# ---- Variedade extra (rodada final): mais frases por pool, para o modelo recombinar menos as mesmas ----
# Regras: não usar "Fortaleza de Cristal"/"Crystal Fortress" nem "Cidadela de Ferro" (são os lugares "novos" dos testes
# em checks.py); todo rumor novo cita o lugar ({pl[...]}), pois a checagem de rumor exige o nome do lugar.
BOOST = {
    "pt": {
        "names": [("Marlow", "m"), ("Vesna", "f"), ("Torvald", "m"), ("Mirel", "f"), ("Orsin", "m"), ("Cassia", "f"),
                  ("Haldor", "m"), ("Nerys", "f"), ("Jorund", "m"), ("Talia", "f"), ("Brenn", "m"), ("Odalys", "f")],
        "places": [("f", "Estalagem do Corvo Cego"), ("m", "Vale das Cinzas Mornas"), ("f", "Cripta dos Sinos Mudos"),
                   ("m", "Farol do Fim do Mundo"), ("f", "Colina dos Três Enforcados"), ("m", "Mercado das Sombras"),
                   ("f", "Floresta dos Sussurros Vermelhos"), ("m", "Porto dos Navios Fantasmas")],
        "omens": ["uma neblina que cheirava a flores de cemitério", "o canto agudo de pássaros que deveriam estar dormindo",
                  "uma sombra comprida demais para a hora do dia", "o rangido de portas que ninguém abria",
                  "um gosto de cinzas na boca", "o som de passos atrás de si, sempre um pouco atrasados"],
        "motives": ["provar a própria inocência", "encontrar o filho de um velho amigo",
                    "devolver um objeto roubado ao dono", "escapar de um contrato assinado sem ler",
                    "cumprir uma promessa feita à beira de um túmulo", "salvar quem um dia o traiu"],
        "comps": ["um velho mercenário de um olho só", "uma menina que desenhava mapas de memória",
                  "um bardo que esquecia todas as letras", "uma coruja que só falava em enigmas",
                  "um cavalo cego que sempre achava o caminho", "uma aprendiz de maga sempre atrasada",
                  "um monge que fizera voto de silêncio", "uma cozinheira com um facão enorme",
                  "um padre que duvidava dos próprios deuses", "um cão de caça meio lobo",
                  "um mapeador que nunca saía do mesmo lugar", "uma bruxa aposentada e mal-humorada",
                  "um minotauro tímido", "uma sombra que seguia o grupo sem ser convidada"],
        "looks": ["cabelos brancos presos em uma trança comprida", "um dente de ouro que brilha ao sorrir",
                  "uma perna de madeira entalhada com runas", "um olho leitoso que parece enxergar demais",
                  "tatuagens de ondas que sobem pelos braços", "uma capa de pele de lobo rasgada na barra",
                  "dedos manchados de ervas", "um sorriso torto que nunca chega aos olhos",
                  "botas grandes demais para os pés", "um anel de ferro que nunca tira",
                  "um cheiro constante de fumaça e cravo", "uma voz grave que treme quando mente"],
        "traits": ["tem medo de altura, mas vive em uma torre", "sempre oferece chá, mesmo no meio de uma briga",
                   "sabe o preço de tudo e o valor de nada", "ri alto demais das próprias piadas",
                   "recusa-se a mentir, mesmo quando seria melhor", "fala com os animais como se fossem velhos amigos",
                   "conta histórias de guerra que nunca viveu", "assobia a mesma melodia o dia inteiro",
                   "desconfia de quem sorri demais", "nunca se senta de costas para a porta",
                   "guarda rancor há mais de trinta anos", "evita pisar em sombras por superstição",
                   "sempre pede desculpas antes de dar más notícias", "coleciona botões de exércitos extintos",
                   "prefere um bom silêncio a uma má conversa", "lembra de todos os nomes e esquece todos os rostos"],
        "secrets": ["tem direito a um trono que todos julgam extinto", "fez um pacto com {th[the]} anos atrás",
                    "já serviu {th[to]} em segredo", "sabe onde {re[the]} está escondido, mas nunca contou",
                    "abriu os portões {pl[of]} na noite do ataque", "vende informações em segredo {th[to]}",
                    "tem um gêmeo que todos julgam morto", "fala a língua dos mortos, mas nunca a usou",
                    "esconde uma marca amaldiçoada sob as roupas", "perdeu um ano inteiro de memória e teme o que fez",
                    "sonha toda noite com {pl[the]} sem nunca ter estado lá",
                    "guarda a última carta de alguém que o grupo procura", "finge não saber ler para ouvir conversas"],
        "npc_hooks": ["Procura voluntários para uma tarefa que paga bem demais.",
                      "Reconhece um membro do grupo e finge que não.",
                      "Pede que o grupo escolte uma carroça sem perguntar o que há nela.",
                      "Sabe onde {th[the]} dorme, mas só conta em troca de proteção.",
                      "Quer comprar {re[the]} por qualquer preço.",
                      "Oferece hospedagem barata, com a condição de que ninguém saia à noite.",
                      "Precisa de ajuda para esconder um corpo antes do amanhecer.",
                      "Tenta recrutar o grupo para uma causa justa demais para ser verdade.",
                      "Conhece um atalho {pl[to]}, mas exige que o grupo jure silêncio.",
                      "Dá ao grupo um aviso enigmático e desaparece."],
        "gm_hooks": ["as portas se fecham sozinhas toda vez que alguém tenta sair",
                     "um teste de Intuição revela que um dos moradores está mentindo",
                     "marcas de garras recentes cobrem as paredes por dentro",
                     "um cadáver recente segura um mapa que aponta para {re[the]}",
                     "os animais da região sempre fogem na mesma direção",
                     "um viajante ferido pede abrigo e sussurra o nome {th[of]}",
                     "o grupo encontra um acampamento abandonado com a comida ainda quente",
                     "uma criança jura ter visto {th[the]} pela janela"],
        "complications": ["o grupo é seguido por alguém que nunca se deixa ver", "o contratante some antes de pagar",
                          "um rival chega primeiro e ofusca a missão", "uma tempestade isola o grupo por três dias",
                          "{th[the]} oferece uma aliança difícil de recusar",
                          "um aliado do grupo carrega uma maldição sem saber", "o mapa está errado em um ponto crucial",
                          "uma criança se agarra ao grupo e se recusa a ir embora"],
        "rewards": ["{n} moedas de prata e uma carta de recomendação de um barão",
                    "um cavalo de guerra e {n} moedas de ouro",
                    "uma poção rara e a promessa de proteção do templo", "um mapa antigo e {n} moedas de ouro",
                    "um anel encantado e a gratidão de uma aldeia inteira",
                    "{n} moedas de ouro e hospedagem grátis na estalagem local"],
        "rumors": ["Dizem que ninguém volta {pl[of]} com o mesmo rosto.",
                   "Um viajante jura que {pl[the]} muda de lugar durante a noite.",
                   "As crianças evitam {pl[the]} depois do pôr do sol.",
                   "Contam que {th[the]} paga em ouro por notícias {pl[of]}.",
                   "Uma viúva afirma que {re[the]} está escondido {pl[in]}.",
                   "Dizem que {comp} entrou {pl[in]} e nunca mais saiu."],
        "surnames": ["Blackmere", "Highcastle", "Wolfsbane", "Emberfall", "Hollowbrook", "Silverfen", "Oakenshield",
                     "Dawnbreaker"],
        "fam_hooks": ["Um dos parentes sumiu há três dias e os outros fingem que está tudo bem.",
                      "A oficina da família pertence, na verdade, {th[to]}.",
                      "Um deles é procurado em outra cidade e usa um nome falso.",
                      "A família guarda {re[the]}, mas ninguém lembra por quê.",
                      "Eles precisam de escolta até {pl[the]} antes da próxima lua cheia.",
                      "Os vizinhos juram ter ouvido a família conversando com os mortos.",
                      "Todos dizem não conhecer {th[the]}, mas ninguém olha nos olhos ao falar.",
                      "Uma herança disputada ameaça separar a família antes do inverno.",
                      "Um dos parentes esconde uma doença grave e pede segredo.",
                      "A família inteira jurou vingança contra {th[the]}, mas ninguém sabe por onde começar."],
    },
    "en": {
        "names": [("Marlow", "m"), ("Vesna", "f"), ("Torvald", "m"), ("Mirel", "f"), ("Orsin", "m"), ("Cassia", "f"),
                  ("Haldor", "m"), ("Nerys", "f"), ("Jorund", "m"), ("Talia", "f"), ("Brenn", "m"), ("Odalys", "f")],
        "places": [("f", "Blind Raven Inn"), ("m", "Valley of Warm Ashes"), ("f", "Crypt of Silent Bells"),
                   ("m", "Lighthouse at the End of the World"), ("f", "Hill of the Three Hanged"),
                   ("m", "Shadow Market"), ("f", "Forest of Red Whispers"), ("m", "Port of Ghost Ships")],
        "omens": ["a fog that smelled of graveyard flowers", "the shrill song of birds that should have been asleep",
                  "a shadow far too long for the hour", "the creak of doors no one was opening",
                  "a taste of ash in the mouth", "footsteps behind, always a little late"],
        "motives": ["prove their own innocence", "find an old friend's missing child",
                    "return a stolen heirloom to its owner", "escape a contract they signed without reading",
                    "keep a promise made at a graveside", "save the one who once betrayed them"],
        "comps": ["a one-eyed old mercenary", "a girl who drew maps from memory", "a bard who forgot every lyric",
                  "an owl that only spoke in riddles", "a blind horse that always found the way",
                  "an apprentice mage who was always late", "a monk who had taken a vow of silence",
                  "a cook with an enormous cleaver", "a priest who doubted his own gods",
                  "a half-wolf hunting dog", "a cartographer who never left the same spot",
                  "a retired and ill-tempered witch", "a shy minotaur", "a shadow that followed the group uninvited"],
        "looks": ["white hair tied in a long braid", "a gold tooth that flashes when they smile",
                  "a wooden leg carved with runes", "a milky eye that seems to see too much",
                  "wave tattoos climbing both arms", "a wolf-fur cloak torn at the hem", "fingers stained with herbs",
                  "a crooked smile that never reaches the eyes", "boots far too big for their feet",
                  "an iron ring they never take off", "a constant smell of smoke and cloves",
                  "a deep voice that trembles when lying"],
        "traits": ["is afraid of heights but lives in a tower", "always offers tea, even in the middle of a brawl",
                   "knows the price of everything and the value of nothing", "laughs too loudly at their own jokes",
                   "refuses to lie, even when it would be better", "talks to animals like old friends",
                   "tells war stories they never lived", "whistles the same tune all day",
                   "distrusts anyone who smiles too much", "never sits with their back to the door",
                   "has held a grudge for thirty years", "avoids stepping on shadows out of superstition",
                   "always apologizes before delivering bad news", "collects buttons from extinct armies",
                   "prefers a good silence to a bad conversation", "remembers every name and forgets every face"],
        "secrets": ["has a claim to a throne everyone believes extinct", "made a pact with {th[the]} years ago",
                    "once served {th[the]} in secret", "knows where {re[the]} is hidden but never told",
                    "opened the gates of {pl[the]} on the night of the attack",
                    "sells information to {th[the]} in secret", "has a twin everyone believes dead",
                    "speaks the language of the dead but has never used it", "hides a cursed mark under their clothes",
                    "lost a whole year of memory and fears what they did",
                    "dreams every night of {pl[the]} without ever having been there",
                    "keeps the last letter of someone the party is looking for",
                    "pretends to be illiterate to overhear conversations"],
        "npc_hooks": ["Is looking for volunteers for a job that pays far too well.",
                      "Recognizes a member of the party and pretends not to.",
                      "Asks the party to escort a wagon without asking what is inside.",
                      "Knows where {th[the]} sleeps, but only tells in exchange for protection.",
                      "Wants to buy {re[the]} at any price.",
                      "Offers cheap lodging on the condition that nobody goes out at night.",
                      "Needs help hiding a body before dawn.",
                      "Tries to recruit the party for a cause too righteous to be true.",
                      "Knows a shortcut to {pl[the]}, but demands an oath of silence.",
                      "Gives the party a cryptic warning and vanishes."],
        "gm_hooks": ["the doors close on their own whenever someone tries to leave",
                     "an Insight check reveals that one of the locals is lying",
                     "fresh claw marks cover the walls from the inside",
                     "a fresh corpse clutches a map pointing to {re[the]}",
                     "the animals of the region always flee in the same direction",
                     "a wounded traveler begs for shelter and whispers the name of {th[the]}",
                     "the party finds an abandoned camp with the food still warm",
                     "a child swears they saw {th[the]} through the window"],
        "complications": ["the party is followed by someone who never shows their face",
                          "the employer vanishes before paying", "a rival arrives first and overshadows the quest",
                          "a storm strands the party for three days",
                          "{th[the]} offers an alliance that is hard to refuse",
                          "one of the party's allies carries a curse without knowing",
                          "the map is wrong at one crucial point",
                          "a child clings to the party and refuses to leave"],
        "rewards": ["{n} silver coins and a letter of recommendation from a baron",
                    "a war horse and {n} gold coins", "a rare potion and the temple's promise of protection",
                    "an old map and {n} gold coins", "an enchanted ring and the gratitude of a whole village",
                    "{n} gold coins and free lodging at the local inn"],
        "rumors": ["They say no one returns from {pl[the]} with the same face.",
                   "A traveler swears {pl[the]} moves around at night.",
                   "Children avoid {pl[the]} after sunset.",
                   "Rumor has it {th[the]} pays in gold for news of {pl[the]}.",
                   "A widow insists {re[the]} is hidden {pl[in]}.",
                   "Folks say {comp} went into {pl[the]} and never came out."],
        "surnames": ["Blackmere", "Highcastle", "Wolfsbane", "Emberfall", "Hollowbrook", "Silverfen", "Oakenshield",
                     "Dawnbreaker"],
        "fam_hooks": ["One of the relatives vanished three days ago and the rest pretend nothing is wrong.",
                      "The family workshop actually belongs to {th[the]}.",
                      "One of them is wanted in another city and uses a false name.",
                      "The family keeps {re[the]}, but no one remembers why.",
                      "They need an escort to {pl[the]} before the next full moon.",
                      "The neighbors swear they heard the family talking to the dead.",
                      "Everyone claims not to know {th[the]}, but no one looks you in the eye when they say it.",
                      "A disputed inheritance threatens to split the family before winter.",
                      "One relative hides a grave illness and begs for silence.",
                      "The whole family swore revenge on {th[the]}, but no one knows where to start."],
    },
}
for _lang, _boost in BOOST.items():
    for _key, _vals in _boost.items():
        PACKS[_lang][_key] = PACKS[_lang][_key] + _vals

SHORT = False   # --short: versões enxutas dos modos (para modelos com contexto de 256 tokens)
LIMIT = None    # (tokenizer, max_tokens) quando --max-tokens é usado


def ctx(P, r):
    name, g = r.choice(P["names"])
    return {
        "h": name, "g": g,
        "cls": P["cls_the"](r.choice(P["classes"]), g),
        "pl": P["np"](*r.choice(P["places"])),
        "th": P["np"](*r.choice(P["threats"])),
        "re": P["np"](*r.choice(P["relics"])),
        "motive": r.choice(P["motives"]), "omen": r.choice(P["omens"]),
        "comp": r.choice(P["comps"]), "tone": r.choice(P["tones"]),
    }


def gen_conto(P, r):
    c = ctx(P, r)
    tm, tf, close = c["tone"]
    vowel = tm[:1].lower() in "aeiou"  # só usado no inglês: a/an
    req = r.choice(P["req_conto"]).format(tm=tm, tf=tf, a="an" if vowel else "a", A="An" if vowel else "A", **c)
    choice = r.choice(P["choices"])
    if SHORT:  # abertura + reviravolta + fecho curto
        # twists[0] e o fecho antigo citam o companheiro, que a versão curta não apresenta
        paras = [r.choice(P["p1"]), r.choice(P["twists"][1:]), r.choice(P["short_endings"])]
        return req, "\n\n".join(fix(p.format(choice=choice, **c)) for p in paras)
    paras = [r.choice(P["p1"]), r.choice(P["p2"]), r.choice(P["twists"]), r.choice(P["endings"])]
    body = "\n\n".join(fix(p.format(choice=choice, **c)) for p in paras) + " " + close
    return req, body


def gen_cenario(P, r):
    c = ctx(P, r)
    req = r.choice(P["req_cen"]).format(**c)
    sim, smell, sound = r.choice(P["sims"]), r.choice(P["smells"]), r.choice(P["sounds"])
    s1, s2, s3 = (fix(t.format(sim=sim, smell=smell, sound=sound, **c)) for t in P["cen"])
    hooks = r.sample(P["gm_hooks"], 1 if SHORT else 2)
    gm = "\n".join(f"- {fix(h.format(**c))}" for h in hooks)
    if SHORT:  # sem a frase dos moradores
        return req, f"{s1} {s2}\n\n{P['cen_gm']}:\n{gm}"
    return req, f"{s1} {s2}\n\n{s3}\n\n{P['cen_gm']}:\n{gm}"


def gen_npc(P, r):
    c = ctx(P, r)
    role = r.choice(P["roles"])[0 if c["g"] == "m" else 1]
    req = r.choice(P["req_npc"]).format(role=role, **c)
    vals = [
        c["h"], role, r.choice(P["looks"]), r.choice(P["traits"]),
        r.choice(P["secrets"]).format(**c), r.choice(P["npc_hooks"]).format(**c),
    ]
    return req, "\n".join(f"{k}: {v}" for k, v in zip(P["npc_labels"], vals))


def gen_missao(P, r):
    c = ctx(P, r)
    lvl, n = r.randint(1, 10), r.choice([50, 100, 150, 200, 300, 500, 800])
    req = r.choice(P["req_quest"]).format(lvl=lvl, **c)
    goal = r.choice(P["goals"]).format(**c)
    vals = [
        r.choice(P["titles"]).format(**c), P["quest_hook"].format(goal=goal), goal,
        fix(r.choice(P["complications"]).format(**c)), r.choice(P["rewards"]).format(n=n),
    ]
    pairs = list(zip(P["quest_labels"], vals))
    if SHORT:  # o Gancho repete o Objetivo; na versão curta sai
        pairs = [pairs[0], pairs[2], pairs[3], pairs[4]]
    return req, "\n".join(f"{k}: {v}" for k, v in pairs)


def gen_rumor(P, r):
    c = ctx(P, r)
    req = r.choice(P["req_rumor"]).format(**c)
    items = r.sample(P["rumors"], 3 if SHORT else 4)
    if not any("{pl" in t for t in items):  # a checagem de rumor exige o nome do lugar do pedido no texto
        items[0] = r.choice([t for t in P["rumors"] if "{pl" in t])
    return req, "\n".join(f"{i}. {fix(t.format(**c))}" for i, t in enumerate(items, 1))


def fam_request(P, r, c):
    """Pedido de família: devolve (pedido, quantidade, tipo). Usado também pelo make_pairs.py."""
    n = r.choice((2, 3, 4))  # uniforme: com "3" mais frequente o modelo aprendia a escrever sempre 3 blocos
    kind = r.choice(P["fam_kinds"])
    n_txt = str(n) if r.random() < 0.3 else P["num_words"][n]  # às vezes com algarismo
    return r.choice(P["fam_req"]).format(n=n_txt, plural=kind[0], **c), n, kind


def gen_familia(P, r):
    c = ctx(P, r)
    req, n, kind = fam_request(P, r, c)
    surname = r.choice(P["surnames"])
    struct = r.choice(P["fam_structs"][n])
    used, members = set(), []
    for rel in struct:
        g = P["fam_rel"][rel]
        name = r.choice([nm for nm, gg in P["names"] if gg == g and nm not in used])
        used.add(name)
        members.append((name, kind[1] if g == "m" else kind[2], rel))
    traits, secrets = r.sample(P["traits"], n), r.sample(P["secrets"], n)  # sem repetir dentro da família
    # Contar é difícil para um modelo pequeno. Dois apoios: (1) o cabeçalho já lista os nomes (a contagem vira "escrever
    # uma lista de n nomes"); (2) cada bloco começa com [i/n], então o modelo sabe quando parar.
    roster = ", ".join(name for name, _occ, _rel in members)
    blocks = []
    for i, ((name, occ, rel), trait, secret) in enumerate(zip(members, traits, secrets), 1):
        vals = [f"{name} {surname}", occ, rel, trait, secret.format(**c)]
        blocks.append(f"[{i}/{n}]\n" + "\n".join(f"{k}: {v}" for k, v in zip(P["fam_labels"], vals)))
    hook = fix(r.choice(P["fam_hooks"]).format(**c))
    head = P["fam_header"].format(s=surname, n=n, roster=roster)
    return req, head + "\n\n" + "\n\n".join(blocks) + f"\n\n{P['fam_hook_label']}: {hook}"


GENS = {"conto": gen_conto, "cenario": gen_cenario, "npc": gen_npc, "missao": gen_missao, "rumor": gen_rumor,
        "familia": gen_familia}
# Mais peso para cenário/missão/rumor: o modelo os confundia com conto ou errava a estrutura.
BASE_COUNTS = {"conto": 500, "cenario": 350, "npc": 250, "missao": 350, "rumor": 300, "familia": 350}


_STOP = {"de", "do", "da", "dos", "das", "o", "a", "os", "as", "um", "uma", "e", "em", "no", "na", "com", "para",
         "por", "que", "ao", "aos", "nos", "nas"}
POEM_REQ = ['Escreva um poema chamado "{t}".', 'Escreva um poema com o título "{t}".', 'Poema: "{t}"']


def from_poems(path, n, rng, per_author=8, min_chars=120, max_chars=1000, min_lines=4):
    """Modo 'poema' (só português): título como pedido, poema como resposta.
    Filtros: poemas curtos o bastante para caber no contexto, com ao menos `min_lines` versos, sem repetição,
    no máximo `per_author` por autor (senão Fernando Pessoa, 14% do CSV, dominaria) e depois sorteio de `n`."""
    import csv
    csv.field_size_limit(10 ** 9)
    # lixo do site de origem: notas, fontes bibliográficas, links, avisos de direitos autorais
    junk = re.compile(r"(^|\n)\s*(in:|nota:|notas?\b|fonte:|extra[ií]d[oa]|retirad[oa] de|tradu[çc][ãa]o|trad\.|"
                      r"publicad[oa]|dispon[ií]vel|veja tamb[eé]m|do livro|do poema|poema d[eo])|"
                      r"https?://|www\.|@\w+\.|©|copyright|isbn|\(\d{4}\)\s*$", re.I)
    seen, by_author, dropped = set(), {}, 0
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            title, text = (r["Title"] or "").strip(), (r["Content"] or "").replace("\r", "").strip()
            text = re.sub(r"[ \t]+\n", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            lines = [l for l in text.split("\n") if l.strip()]
            if not (2 <= len(title) <= 80 and min_chars <= len(text) <= max_chars and len(lines) >= min_lines):
                continue
            tail = lines[-1]  # último verso com ano ou "(in ...)" = assinatura/data/fonte colada ao poema
            if junk.search(text) or junk.search(title) or re.search(r"\b(1[5-9]\d\d|20\d\d)\b", tail) \
                    or re.match(r"\(?\s*in\b", tail, re.I):
                dropped += 1
                continue
            if '"' in title or (title.lower(), text) in seen:
                continue
            # só poemas cujo título tem alguma palavra de conteúdo no corpo: em ~47% dos poemas o título não tem nada a
            # ver com o texto, e o modelo aprendia que o pedido (título) pode ser ignorado
            tw = {w for w in re.findall(r"\w+", title.lower()) if len(w) > 3 and w not in _STOP}
            if not tw & {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3}:
                continue
            seen.add((title.lower(), text))
            by_author.setdefault(r["Author"], []).append((title, text))
    pool = []
    for poems in by_author.values():
        rng.shuffle(poems)
        pool += poems[:per_author]
    rng.shuffle(pool)
    rows = [row("pt", "poema", rng.choice(POEM_REQ).format(t=t), body) for t, body in pool[:n]]
    print(f"poemas: {len(by_author)} autores no filtro, {len(pool)} no pool, {len(rows)} usados "
          f"({dropped} descartados por rodapé/nota/link/direitos)")
    return rows


def row(lang, mode, req, body):
    return {"lang": lang, "mode": mode, "request": req,
            "prompt": build_prompt(lang, mode, req), "completion": body}


def synth(lang, scale, seed):
    P, r = PACKS[lang], random.Random(f"{seed}-{lang}")
    rows, seen = [], set()
    for mode, base_n in BASE_COUNTS.items():
        target, made, tries = int(base_n * scale), 0, 0
        while made < target and tries < target * 30:
            tries += 1
            req, body = GENS[mode](PACKS[lang], r)
            if (req, body) in seen:
                continue
            if LIMIT:
                tok, max_tokens = LIMIT
                n_tok = len(tok(build_prompt(lang, mode, req))["input_ids"]) + len(tok(body)["input_ids"]) + 1
                if n_tok > max_tokens:
                    continue
            seen.add((req, body))
            rows.append(row(lang, mode, req, body))
            made += 1
    return rows


def from_extra(folder, chunk_chars=1400):
    """Textos reais (domínio público etc.): arquivos pt_*.txt / en_*.txt viram exemplos do modo 'continuar'."""
    rows = []
    for f in sorted(pathlib.Path(folder).glob("*.txt")):
        lang = f.name[:2].lower()
        if lang not in PACKS:
            print(f"ignorando {f.name}: nome deve começar com pt_ ou en_")
            continue
        text = re.sub(r"\n{2,}", "\n\n", f.read_text(encoding="utf-8", errors="ignore")).strip()
        buf = ""
        for para in text.split("\n\n"):
            buf += para.strip() + "\n\n"
            if len(buf) >= chunk_chars:
                buf = buf.strip()
                cut = buf.find(". ", 150)
                if 0 < cut < 400:
                    rows.append(row(lang, "conto", PACKS[lang]["extra_req"].format(start=buf[: cut + 1]), buf[cut + 2:]))
                buf = ""
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", default=None, help="pasta com .txt reais (pt_*.txt / en_*.txt)")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument("--poems", default=None, help="CSV de poemas (Author,Title,Content,Views) -> modo 'poema' em PT")
    ap.add_argument("--n-poems", type=int, default=1200, help="quantos poemas usar")
    ap.add_argument("--short", action="store_true", help="versões enxutas dos modos")
    ap.add_argument("--max-tokens", type=int, default=None, help="descarta exemplos maiores que isso (precisa de --tokenizer)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer usado para medir --max-tokens")
    a = ap.parse_args()

    global SHORT, LIMIT
    SHORT = a.short
    if a.max_tokens:
        from transformers import AutoTokenizer

        LIMIT = (AutoTokenizer.from_pretrained(a.tokenizer or "gpt2"), a.max_tokens)

    rows = synth("pt", a.scale, a.seed) + synth("en", a.scale, a.seed)
    if a.poems:
        rows += from_poems(a.poems, a.n_poems, random.Random(f"{a.seed}-poems"))
    if a.extra:
        extra = from_extra(a.extra)
        print(f"extras reais: {len(extra)} exemplos")
        rows += extra
    random.Random(a.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * a.val))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for x in data:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} exemplos -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
