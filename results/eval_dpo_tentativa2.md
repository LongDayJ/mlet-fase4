# Avaliação

Bases: `pt` = pierreguillou/gpt2-small-portuguese, `en` = gpt2

## Perplexidade (validação, só na parte gerada)

| modelo | perplexidade |
|---|---|
| pt · base | 47.53 |
| pt · ajustado | 3.19 |
| pt · dpo | 3.39 |
| en · base | 29.05 |
| en · ajustado | 1.21 |
| en · dpo | 1.38 |

### Perplexidade por modo

Textos de moldes ficam perto de 1; texto natural (poema) é muito menos previsível e puxa a média geral para cima.

| modo | pt · base | pt · ajustado | pt · dpo | en · base | en · ajustado | en · dpo |
|---|---|---|---|---|---|---|
| rumor | 24.20 | 1.24 | 1.37 | 19.59 | 1.25 | 1.76 |
| poema | 99.49 | 68.37 | 70.27 | — | — | — |
| npc | 112.99 | 1.30 | 1.40 | 50.49 | 1.35 | 1.80 |
| conto | 38.23 | 1.17 | 1.25 | 25.52 | 1.17 | 1.26 |
| cenario | 94.12 | 1.16 | 1.24 | 73.04 | 1.18 | 1.32 |
| missao | 34.47 | 1.15 | 1.31 | 29.46 | 1.17 | 1.35 |
| familia | 26.69 | 1.23 | 1.28 | 22.50 | 1.25 | 1.36 |

## Formato correto (% de amostras ok)

Ok = terminou sozinha, tem a estrutura do modo, cita o lugar/personagem do pedido, sem linhas duplicadas. `visto` = lugar que existe no dataset; `novo` = lugar inventado (mede se copia o nome do pedido).

| config | conto/visto | conto/novo | cenario/visto | cenario/novo | npc/visto | npc/novo | missao/visto | missao/novo | rumor/visto | rumor/novo | familia/visto | familia/novo | poema/visto | poema/novo | média |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pt · base | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| pt · ajustado | 97% | 91% | 100% | 100% | 100% | 100% | 100% | 100% | 84% | 94% | 94% | 88% | 100% | 97% | 96% |
| pt · dpo | 100% | 100% | 100% | 100% | 100% | 97% | 41% | 9% | 100% | 100% | 75% | 16% | 56% | 12% | 72% |
| en · base | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | 0% |
| en · ajustado | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 72% | 72% | — | — | 95% |
| en · dpo | 97% | 100% | 88% | 91% | 100% | 100% | 100% | 97% | 100% | 100% | 12% | 12% | — | — | 83% |

## Métricas de geração (diversidade etc.)

| config | distinct-1 | distinct-2 | distinct-3 | repeticao_trigramas | palavras_medias | terminou_sozinho |
|---|---|---|---|---|---|---|
| pt · base | 0.313 | 0.577 | 0.661 | 0.333 | 60.429 | 0.000 |
| pt · ajustado | 0.347 | 0.618 | 0.701 | 0.126 | 75.143 | 1.000 |
| pt · dpo | 0.226 | 0.426 | 0.509 | 0.145 | 104.952 | 0.905 |
| en · base | 0.577 | 0.931 | 0.970 | 0.000 | 153.833 | 0.778 |
| en · ajustado | 0.356 | 0.603 | 0.690 | 0.046 | 70.944 | 1.000 |
| en · dpo | 0.304 | 0.557 | 0.661 | 0.058 | 91.500 | 1.000 |
