# ADR 0003: split agrupado contra quase-duplicatas

- **Status:** aceito
- **Data:** 2026-10-07

## Contexto

O dataset é sintético e cheio de paráfrases do mesmo ticket ("inquiry **about** billing options
for your SaaS..." / "inquiry **regarding** billing options for your SaaS..."). Com um split
aleatório, uma paráfrase vai para o treino e a outra para o teste: a métrica de teste passa a
medir memorização, e não generalização para tickets novos.

No dataset final (14.576 tickets), 8.043 estão em grupos de quase-duplicatas com mais de um
ticket (`reports/data.json`).

## Decisão

1. Construir um grafo ligando tickets com cosseno ≥ **0,8** (TF-IDF `char_wb` 3–5) e usar as
   **componentes conexas** como grupos.
2. Fazer o split treino/validação/teste com **StratifiedGroupKFold**: cada grupo fica inteiro em
   um único split, estratificando por categoria × urgência.
3. Validar a ausência de vazamento no schema Pandera dos splits (check "grupo em mais de um split").

## Justificativa

- Mede o que importa em produção: o desempenho em tickets que o modelo nunca viu, nem parafraseados.
- TF-IDF de caracteres é rápido (o `prepare` inteiro leva ~1,5 min), determinístico e não exige
  modelo neural.
- Componentes conexas tornam a relação transitiva (A~B e B~C ⇒ mesmo grupo).

## Escolha do limiar

`uv run python scripts/near_duplicate_thresholds.py` (subconjunto em inglês do CSV bruto, todas
as 10 filas, 16.338 tickets):

| Limiar | Grupos | Maior grupo | Tickets em grupos com >1 | Grupos com >1 e fila conflitante |
|---|---|---|---|---|
| 0,90 | 14.287 | 5 | 3.570 | 0,0% |
| 0,85 | 12.394 | 6 | 6.636 | 0,4% |
| **0,80** | **10.843** | **14** | **9.027** | **0,6%** |
| 0,75 | 9.725 | 24 | 10.727 | 1,1% |

O tamanho do maior grupo e a fração de grupos com rótulos conflitantes crescem mais rápido abaixo
de 0,8: são cadeias de tickets que já não são paráfrases do mesmo ticket. Em 0,8, só 0,6% dos
grupos misturam filas, o que indica que quase todos são paráfrases genuínas, e ainda assim 9 mil
tickets ficam protegidos contra vazamento.

## Trade-offs aceitos

- O split efetivo é aproximado: StratifiedGroupKFold não aceita fração arbitrária, então 0,15 vira
  1/7 ≈ 0,143 (registrado em `reports/data.json`).
- Paráfrases com muita troca de vocabulário (similaridade < 0,8) ainda podem vazar. Embeddings
  semânticos pegariam mais casos, mas tornariam o pipeline de dados dependente de um modelo neural.
- As métricas ficam mais baixas do que num split aleatório. É o número honesto (ver o efeito
  medido em `docs/experiments.md`).

## Quando revisitar

Se o dataset mudar para dados reais (menos paráfrases) ou se a análise de erros mostrar
paráfrases semânticas vazando entre splits.
