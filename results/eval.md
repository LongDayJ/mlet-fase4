# Avaliação

Bases: `pt` = pierreguillou/gpt2-small-portuguese, `en` = gpt2

## Perplexidade (validação, só na parte gerada)

| modelo | perplexidade |
|---|---|
| pt · base | 50.14 |
| pt · ajustado | 3.34 |
| en · base | 29.05 |
| en · ajustado | 1.21 |

### Perplexidade por modo

Textos de moldes ficam perto de 1; texto natural (poema) é muito menos previsível e puxa a média geral para cima.

| modo | pt · base | pt · ajustado | en · base | en · ajustado |
|---|---|---|---|---|
| rumor | 24.20 | 1.24 | 19.59 | 1.25 |
| poema | 124.18 | 83.80 | — | — |
| npc | 112.99 | 1.30 | 50.49 | 1.35 |
| conto | 38.23 | 1.17 | 25.52 | 1.17 |
| cenario | 94.12 | 1.16 | 73.04 | 1.18 |
| missao | 34.47 | 1.15 | 29.46 | 1.17 |
| familia | 26.69 | 1.23 | 22.50 | 1.25 |

## Formato correto (% de amostras ok)

Ok = terminou sozinha, tem a estrutura do modo, cita o lugar/personagem do pedido, sem linhas duplicadas. `visto` = lugar que existe no dataset; `novo` = lugar inventado (mede se copia o nome do pedido).

| config | conto/visto | conto/novo | cenario/visto | cenario/novo | npc/visto | npc/novo | missao/visto | missao/novo | rumor/visto | rumor/novo | familia/visto | familia/novo | poema/visto | poema/novo | média |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pt · base | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| pt · ajustado | 100% | 92% | 96% | 100% | 100% | 100% | 100% | 100% | 88% | 92% | 96% | 75% | 100% | 100% | 96% |
| en · base | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | 0% |
| en · ajustado | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 75% | 79% | — | — | 96% |

## Métricas de geração (diversidade etc.)

| config | distinct-1 | distinct-2 | distinct-3 | repeticao_trigramas | palavras_medias | terminou_sozinho |
|---|---|---|---|---|---|---|
| pt · base | 0.313 | 0.577 | 0.661 | 0.333 | 60.429 | 0.000 |
| pt · ajustado | 0.347 | 0.618 | 0.701 | 0.126 | 75.143 | 1.000 |
| en · base | 0.620 | 0.962 | 0.996 | 0.000 | 161.933 | 0.733 |
| en · ajustado | 0.378 | 0.605 | 0.680 | 0.048 | 64.467 | 1.000 |

## Varredura de parâmetros (modelo ajustado)

| config | distinct-1 | distinct-2 | distinct-3 | repeticao_trigramas | palavras_medias | terminou_sozinho |
|---|---|---|---|---|---|---|
| pt · T=0.5 rep=1.0 | 0.273 | 0.485 | 0.551 | 0.187 | 88.810 | 1.000 |
| pt · T=0.5 rep=1.05 | 0.308 | 0.540 | 0.615 | 0.179 | 79.095 | 1.000 |
| pt · T=0.5 rep=1.1 | 0.316 | 0.561 | 0.643 | 0.141 | 77.667 | 1.000 |
| pt · T=0.7 rep=1.0 | 0.317 | 0.560 | 0.632 | 0.181 | 79.524 | 1.000 |
| pt · T=0.7 rep=1.05 | 0.347 | 0.618 | 0.701 | 0.126 | 75.143 | 1.000 |
| pt · T=0.7 rep=1.1 | 0.340 | 0.609 | 0.693 | 0.112 | 77.143 | 1.000 |
| pt · T=0.9 rep=1.0 | 0.338 | 0.624 | 0.712 | 0.110 | 78.714 | 1.000 |
| pt · T=0.9 rep=1.05 | 0.366 | 0.662 | 0.744 | 0.071 | 76.714 | 1.000 |
| pt · T=0.9 rep=1.1 | 0.377 | 0.671 | 0.751 | 0.056 | 76.381 | 1.000 |
| pt · T=1.1 rep=1.0 | 0.343 | 0.624 | 0.705 | 0.080 | 79.714 | 1.000 |
| pt · T=1.1 rep=1.05 | 0.360 | 0.654 | 0.725 | 0.067 | 78.143 | 1.000 |
| pt · T=1.1 rep=1.1 | 0.363 | 0.657 | 0.733 | 0.057 | 76.429 | 1.000 |
| en · T=0.5 rep=1.0 | 0.376 | 0.636 | 0.725 | 0.065 | 71.267 | 1.000 |
| en · T=0.5 rep=1.05 | 0.387 | 0.637 | 0.718 | 0.051 | 66.867 | 1.000 |
| en · T=0.5 rep=1.1 | 0.386 | 0.636 | 0.731 | 0.027 | 64.867 | 1.000 |
| en · T=0.7 rep=1.0 | 0.388 | 0.647 | 0.735 | 0.043 | 70.667 | 1.000 |
| en · T=0.7 rep=1.05 | 0.378 | 0.605 | 0.680 | 0.048 | 64.467 | 1.000 |
| en · T=0.7 rep=1.1 | 0.403 | 0.658 | 0.753 | 0.033 | 68.533 | 1.000 |
| en · T=0.9 rep=1.0 | 0.388 | 0.655 | 0.747 | 0.069 | 71.467 | 1.000 |
| en · T=0.9 rep=1.05 | 0.394 | 0.656 | 0.743 | 0.045 | 70.000 | 1.000 |
| en · T=0.9 rep=1.1 | 0.390 | 0.657 | 0.754 | 0.026 | 74.000 | 1.000 |
| en · T=1.1 rep=1.0 | 0.405 | 0.685 | 0.772 | 0.068 | 71.200 | 1.000 |
| en · T=1.1 rep=1.05 | 0.403 | 0.658 | 0.743 | 0.046 | 67.333 | 1.000 |
| en · T=1.1 rep=1.1 | 0.415 | 0.709 | 0.805 | 0.025 | 72.067 | 1.000 |
