# ADR 0005: como a API carrega e escala os modelos

- **Status:** aceito
- **Data:** 2026-10-07

## Contexto

A API de triagem (Fase 2) precisa carregar os campeões do Model Registry. Mas o registry é um
SQLite local com artefatos em disco (ADR 0004): um container ou um PaaS (Render, Railway) não
enxerga esses arquivos. Além disso, a inferência é CPU-bound, e era preciso decidir como escalar.

## Decisão

1. **Uma URI de modelo configurável por alvo** (`API_MODEL_URI_CATEGORY`/`_URGENCY`):
   - em desenvolvimento: `models:/supportai-triage-<alvo>@champion` (registry);
   - em container/deploy: um diretório exportado por `python -m supportai.classifier.export`,
     com um `registry.json` (nome, versão, run) para a API reportar exatamente o que serve.
2. **Carregamento no startup** (lifespan do FastAPI): se um modelo faltar, o processo não sobe,
   em vez de falhar na primeira requisição.
3. **Escala por processos, não por threads:** 1 worker uvicorn por container e réplicas
   horizontais (ou `--workers N` numa máquina só).
4. **Imagem com dois alvos:** `runtime` (modelos montados por volume) e `runtime-with-models`
   (modelos embutidos, para PaaS sem volume).

## Justificativa

- A troca de modelo continua sendo "mover o alias + exportar + deploy", com a versão visível em
  `/health`, nos logs e na métrica `supportai_model_info`.
- Medido em `scripts/benchmark_latency.py` (4 vCPUs, 1.000 requisições):

  | Workers | Concorrência | p50 | p95 | p99 | Throughput |
  |---|---|---|---|---|---|
  | 1 | 1 | 9,1 ms | 13,6 ms | 16,6 ms | 102 req/s |
  | 1 | 4 | 47,6 ms | 68,9 ms | 80,6 ms | 83 req/s |
  | 4 | 4 | 12,1 ms | 28,5 ms | 36,5 ms | 261 req/s |

  Com 1 processo, mais concorrência **piora** o throughput: a tokenização do TF-IDF é Python
  puro e segura o GIL, então as threads do FastAPI só disputam a CPU. Com 4 processos, o
  throughput vai a 3,1×.

## Trade-offs aceitos

- **Memória:** cada worker ocupa ~380 MB de RSS com os dois modelos carregados
  (`uv run python scripts/api_memory.py`: 163 MB após os imports, 356 MB com o primeiro modelo,
  383 MB com o segundo). Quase tudo é biblioteca: o primeiro carregamento puxa skops, scipy e a
  pilha do MLflow, e o segundo modelo acrescenta só ~27 MB. Num plano de 512 MB cabe 1 worker,
  com folga pequena.
- O modelo exportado é uma cópia: se o alias mudar no registry, o container só vê a mudança no
  próximo build. É intencional: o que está em produção é imutável e rastreável pela imagem.
- `API_ALLOW_MODEL_DESERIALIZATION=true` está ligado na imagem, porque ela só carrega modelos
  que nós mesmos exportamos.

## Quando revisitar

- Se a memória apertar: carregar o `.skops` direto, sem a pilha do MLflow, e usar `mlflow-skinny`
  na imagem de serviço.
- Com um registry remoto (servidor MLflow + object storage), a API pode carregar
  `models:/...@champion` direto, sem exportar.
