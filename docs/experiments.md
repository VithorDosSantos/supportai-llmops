# Experimentos

Regra do projeto: **todo número aqui vem de uma execução real**, com o comando para reproduzi-la,
o commit e o run do MLflow correspondente.

## Fase 1: classificador de triagem

### Dados

`uv run dvc repro prepare` → [`reports/data.json`](../reports/data.json)

| Etapa | Linhas |
|---|---|
| CSV bruto (inglês + alemão) | 28.587 |
| Só inglês | 16.338 |
| 6 filas principais | 14.576 (após remover 1 ticket curto demais) |
| Treino / validação / teste | 10.411 / 2.083 / 2.082 (71,4% / 14,3% / 14,3%) |

A distribuição de categoria e urgência ficou praticamente idêntica entre os splits (diferença
máxima de 0,1 ponto percentual; ver `category_distribution` em `reports/data.json`). 8.043
tickets estão em grupos de quase-duplicatas com mais de um ticket; o maior grupo tem 11.

### Protocolo

- **Seleção** do hiperparâmetro C (regressão logística) pelo F1 macro na **validação**.
- **Teste** avaliado uma única vez por run, com o C escolhido.
- **Campeão** do Model Registry escolhido pela validação (`classifier/registry.py`).
- `class_weight="balanced"` em todos os modelos; seed 42.

### Resultados (teste)

| Modelo | Alvo | C | F1 macro val | **F1 macro teste** | Acurácia teste | Run MLflow |
|---|---|---|---|---|---|---|
| TF-IDF (palavras 1-2 + caracteres 3-5) + LogReg | categoria | 64 | 0,432 | **0,434** | 0,446 | `0141adfdc2a04f20bb53e25fdc388b2f` |
| TF-IDF (palavras 1-2 + caracteres 3-5) + LogReg | urgência | 16 | 0,476 | **0,478** | 0,509 | `40a9e19b15744101a19923615815a60f` |
| paraphrase-multilingual-MiniLM-L12-v2 + LogReg | ambos | | | *pendente* | | |
| multilingual-e5-small + LogReg | ambos | | | *pendente* | | |

Comando: `uv run dvc repro train_tfidf` (commit `7c33006`). Resumo em
[`reports/train_tfidf.json`](../reports/train_tfidf.json); matrizes de confusão, relatórios por
classe e erros do teste ficam nos artefatos `evaluation/` de cada run do MLflow.

Grid de C na validação (categoria): 1 → 0,410; 4 → 0,420; 16 → 0,430; 64 → 0,432. O ganho já
estava em platô; o melhor C ficou na borda do grid, mas ampliar o grid traria ganho marginal.

### Por classe: TF-IDF, categoria (teste)

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| billing_payments | 0,770 | 0,678 | **0,721** | 227 |
| technical_support | 0,481 | 0,507 | 0,494 | 677 |
| product_support | 0,386 | 0,428 | 0,406 | 439 |
| customer_service | 0,382 | 0,380 | 0,381 | 345 |
| it_support | 0,324 | 0,299 | 0,311 | 278 |
| returns_exchanges | 0,349 | 0,250 | **0,291** | 116 |

Principais confusões (linha = real, coluna = previsto): `it_support → technical_support` (109 de
278), `product_support → technical_support` (131 de 439), `technical_support → product_support`
(133 de 677). As três filas técnicas se sobrepõem muito no texto.

### Por classe: TF-IDF, urgência (teste)

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| high | 0,575 | 0,575 | 0,575 | 817 |
| medium | 0,520 | 0,533 | 0,527 | 865 |
| low | 0,340 | 0,323 | 0,331 | 400 |

### Por que os números são baixos

Não é (só) o modelo: o sinal nos rótulos é fraco.

```
$ uv run python scripts/label_keywords.py
                   return/exchange/refund  billing/invoice/charge/payment
billing_payments                    0.017                           0.591
customer_service                    0.009                           0.000
it_support                          0.007                           0.000
product_support                     0.007                           0.000
returns_exchanges                   0.120                           0.005
technical_support                   0.007                           0.000
```

Só 12% dos tickets de `returns_exchanges` mencionam devolução, troca ou reembolso; a maioria fala
de integrações de software, relatórios etc. Já `billing_payments`, a única fila com vocabulário
próprio (59% citam cobrança ou pagamento), é a de maior F1. O teto deste dataset é baixo para
qualquer modelo de texto, o que reforça o uso do F1 para **comparar** abordagens, e não como
estimativa de produção.

### Efeito do split agrupado

_Em execução._
