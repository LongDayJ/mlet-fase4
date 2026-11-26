# 🐉 Forja de Contos — gerador de fantasia para RPG (GPT-2 + LoRA)

Playground em Streamlit que gera **contos, cenários, NPCs, famílias de NPCs, missões e rumores de taverna** para RPG de
fantasia, em **português ou inglês**, e **poemas / canções de bardo** em português. Projeto da Fase 4 de Machine Learning
Engineering.

## Estratégia

| Etapa | Decisão |
|---|---|
| Tema | Conteúdo de fantasia para mestres de RPG, em 7 modos (conto, cenário, NPC, **família de 2 a 4 NPCs**, missão, rumores e poema só em PT) |
| Modelos base | **Um por idioma** (ambos GPT-2 small, 124M): PT = `pierreguillou/gpt2-small-portuguese`, EN = `gpt2`. Leves o bastante para CPU |
| Ajuste | **LoRA** (peft) em todas as camadas lineares: um adaptador de ~19 MB por idioma (`adapter/pt`, `adapter/en`), cada um treinado só com os exemplos do seu idioma, 7 épocas |
| Dataset | 9.600 exemplos: 4.200 sintéticos por idioma (moldes com muitas combinações, `data/build_dataset.py`) + 1.200 poemas reais em PT |
| Idioma | O usuário escolhe no app; o app carrega só o modelo daquele idioma (o anterior sai da memória) |
| Avaliação | Por idioma: perplexidade (geral e **por modo**), **formato correto** (estrutura do modo, cita o lugar do pedido, termina sozinho, nº exato de NPCs; inclui pedidos com **lugar inventado**), distinct-n, repetição de trigramas e varredura de temperatura/penalidade |
| Uso | O app **confere o formato** do texto gerado com a mesma checagem da avaliação e gera de novo (até 4 vezes) se falhar |
| Deploy | Streamlit; o app baixa a base do Hugging Face e aplica o adaptador que está neste repositório. O modelo não é publicado no Hub |

Formato de treino (a perda só conta na resposta). Os rótulos ficam em inglês nos dois idiomas; só o pedido e o texto mudam:

```
### Language: pt
### Mode: npc
### Request: Crie um NPC: ferreira no Porto de Âmbar.
### Text:
Nome: Lysa
Ocupação: ferreira
...
```

## Resultados (modelo final, 24 amostras por pedido)

Perplexidade na validação (só na parte gerada): PT 50,1 (base) → 3,34 (ajustado); EN 29,1 → 1,21.

**Atenção:** a perplexidade perto de 1 nos modos de molde **não mede qualidade**: o texto é gerado por moldes e é previsível por
construção. Por isso a métrica principal é o formato correto. A perplexidade por modo (`results/eval.md`) mostra que os
modos de molde ficam entre 1,15 e 1,35 e que só o **poema** (texto natural) é alto (83,8): é ele que puxa a média do PT.

Formato correto (% de amostras ok; "visto" = lugar que existe no dataset, "novo" = lugar inventado):

| Modelo | média | conto | cenário | NPC | missão | rumor | família (visto / novo) | poema |
|---|---|---|---|---|---|---|---|---|
| PT base | 0% | 0% | 0% | 0% | 0% | 0% | 0% / 0% | 0% |
| PT ajustado | **96%** | 92–100% | 96–100% | 100% | 100% | 88–92% | 96% / 75% | 100% |
| EN base | 0% | 0% | 0% | 0% | 0% | 0% | 0% / 0% | — |
| EN ajustado | **96%** | 100% | 100% | 100% | 100% | 100% | 75% / 79% | — |

Com 24 amostras por pedido, a margem de erro é de cerca de ±9 pontos: diferenças de poucos pontos entre linhas não são
reais. O modelo base nunca passa porque não termina sozinho nem segue a estrutura (ele não sabe o formato do prompt).
Tabelas completas, diversidade e varredura de parâmetros: `results/eval.md`; textos lado a lado: `results/samples.md`.

## O app confere e tenta de novo

Um GPT-2 pequeno às vezes erra a estrutura (um NPC a menos, nome repetido). Na barra lateral, **"Conferir o formato e tentar
de novo"** (ligado por padrão) roda `checks.check_request` em cada texto e, se falhar, gera aquela versão de novo, até 4
vezes. Medido em 12 pedidos de família (3 e 4 NPCs, PT e EN): de **8–10 de 12 na primeira tentativa para 12 de 12** em até
4 tentativas. Cada texto mostra se passou (e em qual tentativa) ou o motivo da falha. Poema não tem checagem.

## Modo "família de NPCs" (ensinar o modelo a contar)

O GPT-2 ajustado não segue instruções: ele completa padrões. Pedir "três NPCs" a um modelo treinado só com NPCs avulsos
devolve um NPC. Para o modelo contar, o dataset ganhou o modo `familia`: pedidos como
`Crie uma família de três NPCs: anões para o Mithral Hall.` com respostas de 2 a 4 NPCs, mesmo sobrenome, papéis
(pai, mãe, filha, tio...) e um gancho da família. A avaliação confere o número exato de NPCs, o sobrenome comum, nomes
diferentes e os campos.

**Contar é o difícil.** Na 1ª versão (cabeçalho `Família X (3 membros)` e depois os blocos) o modelo acertou todo o formato mas
errou a contagem: só 25% (pedido "visto") e 0% ("novo") das famílias saíram com o número certo de NPCs. Na 2ª versão o formato
ajuda a contar: o cabeçalho já traz a lista de nomes (`Família Valdris (3 membros): Brisa, Fenrick, Thalion`) e cada bloco
começa com `[i/n]`. O número de membros passou a ser uniforme entre 2, 3 e 4. Resultado: 83–88% no PT.

**Variedade.** Com os moldes iniciais, o gancho da família era sorteado entre 6 frases e os segredos entre ~30, e o modelo
repetia as mesmas ("uma mula teimosa chamada Pimenta" aparecia em ~1 de cada 6 famílias). Na versão final cada pool tem de 3 a 5
vezes mais frases (segredos distintos: ~30 → 121; ganchos: 6 → 140), e a "Pimenta" aparece em 4 de 660 famílias. Na
avaliação final o modo melhorou (PT 96% / 75%; EN, que estava em 38% / 8%, foi para 75% / 79%).

**Limite honesto:** ele aprende **esse** pedido (quantidade 2–4, tipos de NPC do dataset). Variações livres continuam fora do
que ele sabe, e o texto continua sendo recombinação de moldes: mais variedade de peças, não criatividade nova.

**Sobre as checagens:** só se exige no texto o que o dataset sempre contém. Por isso o nome do lugar não é exigido em NPC e
missão (aparece em parte dos exemplos). Validamos a régua rodando-a sobre o próprio dataset: 100% dos exemplos passam.

## Modo "poema / canção de bardo" (só português)

Sétimo modo, só no adaptador PT: `Escreva um poema chamado "A Torre do Corvo".` O título é o pedido e o poema é a resposta.
Vem do dataset [Poems in Portuguese](https://www.kaggle.com/datasets/oliveirasp6/poems-in-portuguese)
(Kaggle, 15.543 poemas copiados de escritas.org; licença declarada na página: GPL 2). Uso: 1.200 poemas depois de filtrar
poemas muito longos ou curtos, com rodapé/nota/link/direitos autorais ou data/fonte no último verso, com no máximo 8 por
autor (Fernando Pessoa tem 14% do CSV) e sem duplicatas.

**Filtro título → texto.** Em 47% dos poemas o título não tinha nenhuma palavra em comum com o texto (é só um rótulo do site),
e o modelo aprendia que o pedido pode ser ignorado. Agora só entram poemas em que alguma palavra de conteúdo do título aparece
no poema (100% dos usados).

```bash
python data/build_dataset.py --scale 2 --poems data/poems/portuguese-poems.csv
```

**O CSV e os `.jsonl` gerados não entram no repositório** (`.gitignore`): baixe o CSV no Kaggle e coloque em `data/poems/`.
Os autores incluem nomes ainda sob direitos autorais; o modelo é pequeno e pode reproduzir trechos, por isso não há garantia
de que o texto gerado seja inédito.

**Palavras impróprias nos poemas (limitação conhecida).** Os poemas são texto de terceiros e **não foram filtrados contra a
lista de palavras bloqueadas** (`BLOCKED_PT`/`BLOCKED_EN` em `common.py`). Ao conferir o dataset gerado, 18 dos 9.120 exemplos de
treino (e 1 dos 480 de validação) tinham palavra bloqueada, todos poemas (palavrões como "puta", "porra", "vadia"). O
modelo PT viu esses poemas. Decidimos não retreinar e mitigar só na ponta: o app recusa pedidos com essas palavras
(`blocked_in`) e `engine.py` proíbe o modelo de gerá-las. Se esse filtro falhar, o modelo pode reproduzi-las. A correção
completa seria filtrar os poemas com `blocked_in` no `build_dataset.py` e retreinar o adaptador PT. Os modos de moldes
(conto, cenário, NPC, missão, rumor, família) e todo o inglês não são afetados, pois o vocabulário vem de listas escritas à mão.

**Limite do poema:** o modelo aprende a *forma* de poema (versos curtos, tom lírico), não o *assunto*. Pedir "A cobra e a
Águia" gera um poema lírico genérico, com palavras do título aparecendo de vez em quando, não um poema sobre uma cobra e uma
águia. A checagem automática (≥4 versos, sem vazar o formato) dá 100%, mas é frouxa e **não mede se o poema fala do tema**. Um
GPT-2 de 124M com 1.200 poemas não vai além disso. No app a penalidade de repetição do poema abre em 1,15 (com 1,05 alguns
poemas entravam em loop de refrão).

## Por que dois modelos base

Um teste rápido (mesmos prompts, uma amostra por idioma) mostrou que nenhum GPT-2 serve bem aos dois idiomas:

| Base | Prompt em PT | Prompt em EN |
|---|---|---|
| `gpt2` | ignora o português e continua em inglês | coerente |
| `pierreguillou/gpt2-small-portuguese` | coerente | quebrado (escorrega para PT) |
| `yanH2003/fine-tuned-gpt2-portuguese` | coerente | sem sentido |

O `yanH2003` é um ajuste do `pierreguillou` sem dados documentados e sem ganho visível, então usamos o `pierreguillou` direto
(dá para trocar com `BASE_PT=yanH2003/fine-tuned-gpt2-portuguese`).

Também foi testado e descartado `zhzh98/gpt2-portuguese20k-english10k-configA-1` (bilíngue, 92,9M): contexto de apenas
256 tokens (53% dos exemplos estouravam), tokenizer que só funciona carregando o `tokenizer.json` direto (via
`AutoTokenizer` descarta os espaços) e, com o tokenizer correto, continua prompts em português **em inglês**.

## DPO (tentativa 1, resultado negativo): aprendizado por preferência em cima do LoRA

Segunda etapa testada depois do SFT: o mesmo adaptador LoRA refinado com **DPO** (Direct Preference Optimization), em PyTorch
puro (`train_dpo.py`, ~60 linhas, sem a biblioteca `trl`).

1. `make_pairs.py`: o modelo SFT escreve 8 versões de cada pedido em temperatura alta; o `checks.py` marca quais passam
   (termina sozinha, estrutura do modo, cita lugar/herói, sem linhas duplicadas). Cada par = uma versão que passa (*chosen*)
   contra uma que falha (*rejected*) para o mesmo pedido. 30% dos pedidos usam lugares inventados.
2. `train_dpo.py --lang pt|en`: perda `-log σ(β[(logp_chosen − logp_rejected) − (ref_chosen − ref_rejected)])`, em que a
   referência é o próprio SFT congelado (probabilidades calculadas antes do treino). Resultado: `adapter_dpo/<lang>`.
3. `evaluate.py` compara automaticamente **base × ajustado (SFT) × dpo** quando `adapter_dpo/` existe; o app ganha a opção
   "SFT + DPO".

**Resultado:** na primeira tentativa o DPO **não trouxe ganho líquido de formato** e **reduziu a diversidade** (distinct-n
caiu 6–7%). Por isso o modelo final do app é o SFT. As limitações eram esperadas: a "recompensa" é uma checagem automática de
formato, não de qualidade literária, e o DPO tende a reduzir a diversidade. Em vez de insistir no DPO, o ganho de formato veio
de duas coisas mais baratas: o formato de dados da família (lista de nomes + `[i/n]`) e a conferência com nova tentativa no app.

### Tentativa 2 (pares equilibrados, 32 amostras por pedido): piorou, e o SFT continua sendo o modelo do app

Rodamos o DPO de novo sobre o SFT final (`colab_dpo.ipynb`): ~20 pares por idioma e modo (108 pares de treino em PT, 98 em EN),
avaliação com 32 amostras por pedido. Com lr 1e-5 e 1 época o ajuste não saiu do lugar (loss 0,69 = `ln 2`, margem ~0). Com uma
taxa maior e 10 épocas ele decorou os pares de treino (acerto 100%, margem +7) mas não generalizou: **acerto na validação de
50%, o acaso**. Avaliação do adaptador DPO final contra o SFT:

| | SFT | DPO |
|---|---|---|
| PT, formato correto (média) | **96%** | 72% |
| EN, formato correto (média) | **95%** | 83% |
| PT, distinct-2 (diversidade) | **0,618** | 0,426 |
| EN, distinct-2 | **0,603** | 0,557 |
| PT, terminou sozinho | **100%** | 90% |
| Família PT / EN (média visto + novo) | **91% / 72%** | 46% / 12% |

O DPO melhorou rumores e contos em PT (84–94% → 100%) mas **quebrou** missão, poema e família, e reduziu a diversidade. Por que não
funcionou: (1) poucos pares (~100 por idioma, de 6 a 7 modos) para um ajuste de 4,7 milhões de parâmetros; (2) as falhas
rejeitadas são de tipos diferentes (não terminou, faltou campo, NPC a menos), então não há um padrão único a aprender; (3) a
recompensa só mede formato, e o SFT já acerta ~96%, então sobra pouco a ganhar. **Conclusão: o app usa o SFT**; o DPO
fica documentado como resultado negativo. O que de fato elevou o formato foi o desenho dos dados (lista de nomes + `[i/n]` na
família) e a conferência com nova tentativa no app.

## Parâmetros de geração

Os padrões do app (temperatura 0,7; penalidade de repetição 1,05; top-p 0,92; top-k 50) vieram de um teste com checagem automática
de formato (4 amostras por modo e idioma, 10 pedidos): com temperatura 0,9 e penalidade 1,1 só **21 de 40** amostras saíam
corretas; com 0,7 e 1,05, **33 de 40**. A penalidade de repetição alta é prejudicial aqui porque penaliza também os tokens
já presentes no pedido, como o nome do lugar que o modelo deveria copiar. Temperaturas mais altas dão mais variedade ao custo
de mais erros (nomes trocados, estrutura quebrada); o playground deixa ajustar tudo. Exceção: o poema usa 1,15.

## Como reproduzir

### 1. Dataset (local, segundos)
```bash
python data/build_dataset.py --scale 2 --poems data/poems/portuguese-poems.csv   # data/train.jsonl e val.jsonl (9.600 exemplos)
```
Sem `--poems` o dataset sai sem o modo poema. Textos reais: coloque arquivos `pt_*.txt` / `en_*.txt` (domínio público, por
exemplo) em `data/extra/` e rode com `--extra data/extra`.

### 2. Treino (Google Colab, GPU T4)
Use o notebook `colab_train.ipynb` (Ambiente de execução → Alterar tipo → **T4 GPU**; ele pede o zip do projeto com o CSV de poemas).
Ou, em células próprias:
```python
!pip install -q transformers peft datasets accelerate
!pip uninstall -y -q torchao   # o Colab traz um torchao antigo que o peft novo recusa
!python data/build_dataset.py --scale 2 --poems data/poems/portuguese-poems.csv
!python train.py --lang pt --epochs 7   # -> adapter/pt  (~14 min na T4)
!python train.py --lang en --epochs 7   # -> adapter/en
!python evaluate.py --sweep --n-format 24   # results/eval.md e results/samples.md
```
Baixe as pastas `adapter/` e `results/` e coloque no repositório. Faça o backup dos adaptadores **antes** da avaliação: se o Colab
reiniciar, o treino se perde.

O treino salva um ponto de retomada a cada 25 passos. Se for interrompido, basta rodar o **mesmo comando** de novo: ele continua
de onde parou e pula o idioma que já terminou (`--force` para retreinar do zero). Mantenha os mesmos `--epochs`, `--batch` e `--lr`.

### 3. App local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sem a pasta `adapter/<idioma>`, o app roda com o modelo base original daquele idioma (e avisa).

### 4. Deploy (Streamlit Community Cloud)
1. Suba o repositório (inclua `adapter/pt` e `adapter/en`, ~19 MB cada; **não** suba `data/poems/` nem `data/*.jsonl`).
2. share.streamlit.io → New app → escolha o repositório e `app.py`.
3. O app usa cerca de 1,0–1,1 GB de RAM com um idioma carregado. Se estourar a memória, o plano B é um Space do Hugging Face com SDK Streamlit.

## Limitações

- **Originalidade:** os textos do dataset vêm de moldes; o modelo recombina peças e aprende estrutura, não escreve nada
  realmente novo. A variedade de frases foi ampliada, mas o limite continua.
- **Contar e nomes:** a família acerta ~75–96% (depende do pedido). Erros ocasionais (nome repetido, um NPC a menos) são
  corrigidos pela conferência do app, que tenta de novo.
- **Poema:** forma sem assunto (ver acima) e possível reprodução de trechos de autores reais.
- **Escopo:** ele aprende os pedidos no estilo dos exemplos. Instruções livres não são seguidas.
- **Perplexidade:** perto de 1 nos modos de molde não é qualidade, só previsibilidade.

## Arquivos
- `common.py` — formato do prompt, modos, exemplos, bases por idioma
- `engine.py` — carregar modelo e gerar texto
- `train.py` — fine-tuning LoRA (`--lang pt|en`)
- `evaluate.py` — métricas e comparação base × ajustado, por idioma
- `checks.py` — checagens automáticas de formato (avaliação, pares do DPO e conferência do app)
- `make_pairs.py` — gera os pares de preferência (boa × ruim) para o DPO
- `train_dpo.py` — DPO sobre o adaptador LoRA (`--lang pt|en`)
- `app.py` — playground
- `data/build_dataset.py` — gerador do dataset
- `colab_train.ipynb`, `colab_dpo.ipynb` — notebooks de treino e do DPO
- `results/` — `eval.md`, `eval.json`, `samples.md`, métricas de treino
- `entrega.txt` — links da entrega
