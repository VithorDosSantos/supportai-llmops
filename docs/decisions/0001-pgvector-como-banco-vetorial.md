# ADR 0001: pgvector como banco vetorial

- **Status:** aceito
- **Data:** 2026-10-06

## Contexto

O RAG (Fase 3) precisa de busca vetorial, busca lexical (BM25/full-text) e metadados
(fonte, seção) para citar os documentos. Também vamos guardar tickets e, possivelmente,
resultados de avaliação. As alternativas consideradas foram **pgvector**, **Qdrant** e
**Chroma**.

## Decisão

Usar **PostgreSQL + pgvector**.

## Justificativa

- **Um banco só:** vetores, texto, metadados e dados transacionais no mesmo lugar. Filtros por
  metadados e joins são SQL comum, sem sincronizar dois sistemas.
- **Busca híbrida nativa:** o full-text search do Postgres (`tsvector`, com dicionário `portuguese`)
  roda ao lado do pgvector, então dá para fazer a fusão RRF numa única query.
- **Operação:** Postgres gerenciado existe em praticamente toda cloud (e no free tier de Render e
  Railway). É uma tecnologia que qualquer time de engenharia já opera.
- **Escala suficiente:** a base de conhecimento tem dezenas de documentos (milhares de chunks).
  O índice HNSW do pgvector atende com folga.

## Trade-offs aceitos

- Para centenas de milhões de vetores, ou com muita escrita concorrente, bancos dedicados (Qdrant,
  Milvus) têm melhor desempenho e recursos como quantização e sharding nativos.
- O ranking do full-text do Postgres (`ts_rank`) não é BM25 de verdade. Se isso afetar o recall
  medido na Fase 4, a alternativa é calcular o BM25 em memória (`rank-bm25`), o que funciona bem
  no tamanho atual da base.

## Quando revisitar

Se a base passar de ~10M de vetores ou se a latência p95 do retrieval ultrapassar o orçamento
definido na Fase 2.
