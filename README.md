# SupportAI: atendimento ao cliente com ML, RAG avaliado e LLMOps

[![CI](https://github.com/VithorDosSantos/supportai-llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/VithorDosSantos/supportai-llmops/actions/workflows/ci.yml)

Sistema que recebe tickets de suporte da **Voltix**, uma loja online fictícia de eletrônicos e assinaturas, e:

1. **Triagem (ML clássico):** classifica a categoria (6 filas: suporte técnico, suporte de produto, atendimento, TI, cobrança, devoluções/trocas) e a urgência (baixa, média, alta).
2. **Resposta sugerida (RAG):** busca na base de conhecimento e gera uma resposta **citando as fontes**, ou recusa quando não há informação suficiente.
3. **Agente (opcional):** consulta pedidos e solicita reembolsos numa API simulada, com aprovação humana para ações sensíveis.

Os tickets e a base de conhecimento estão em **inglês** (ver [ADR 0002](docs/decisions/0002-dataset-publico-sintetico-em-ingles.md)); a documentação está em português.

O foco não é só "funcionar". O projeto ataca problemas comuns de levar ML/LLM para produção: reprodutibilidade, avaliação automatizada de RAG, regressão de prompts no CI, custo e latência, PII/LGPD, observabilidade e drift.

> **Status:** Fase 2 (API em produção) implementada. Na Fase 1, falta só a execução real do modelo de embeddings (código e testes prontos). Todo número publicado aqui vem de uma execução real, com o comando para reproduzi-lo.

## Roadmap

| Fase | Conteúdo | Status |
|---|---|---|
| 0 | Setup: estrutura, tooling, Postgres + pgvector, config, CI | ✅ |
| 1 | Dados + classificador de triagem (TF-IDF vs. embeddings, MLflow, DVC) | 🚧 |
| 2 | Servir em produção: FastAPI, Docker, benchmark de latência | ✅ |
| 3 | RAG: chunking, busca híbrida, reranking, citações, recusa | ⏳ |
| 4 | Avaliação: retrieval + geração, regressão no CI | ⏳ |
| 5 | Custo, cache semântico, roteamento, PII, prompt injection, tracing, drift | ⏳ |
| 6 | Agente com tool calling e human-in-the-loop (opcional) | ⏳ |
| 7 | Demo Streamlit, ADRs, case de negócio | ⏳ |

## Dados

| | |
|---|---|
| Fonte | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face), licença **CC-BY-NC 4.0** |
| Recorte | Só inglês; 6 filas mais frequentes (~89% dos tickets em inglês) |
| Rótulos | Categoria = `queue`; urgência = `priority` em 3 níveis |
| Tamanho | 14.576 tickets após a limpeza: 10.411 treino / 2.083 validação / 2.082 teste |
| Reprodutibilidade | Revisão do repositório fixada + SHA-256 do CSV + `dvc.lock` |

**Limitações, ditas com honestidade:**

- **O dataset é sintético** (o próprio autor o descreve como gerado por um "Synthetic IT Ticket Generator"). Não representa a distribuição de um e-commerce real, e os tickets falam mais de SaaS/TI corporativa do que de compras.
- **Muitas paráfrases.** 8.043 dos 14.576 tickets estão em grupos de quase-duplicatas (cosseno ≥ 0,8). Por isso o split é **agrupado**: paráfrases do mesmo ticket ficam no mesmo split ([ADR 0003](docs/decisions/0003-split-agrupado-contra-quase-duplicatas.md)). Com um split aleatório, o mesmo modelo marcaria F1 macro 0,730 em vez de 0,445 na categoria: **+0,285 de inflação** (`uv run python scripts/split_leakage.py`; esse experimento treina com treino + validação, por isso 0,445 e não o 0,434 da tabela abaixo).
- **Rótulos ruidosos.** Só 12% dos tickets da fila `returns_exchanges` mencionam "return", "exchange" ou "refund" (`uv run python scripts/label_keywords.py`). Isso limita o F1 de qualquer modelo: os números absolutos valem para comparar abordagens, não como estimativa de desempenho em produção.
- **Licença não comercial.** Adequada a portfólio; uso comercial exigiria outro dataset.

A base de conhecimento (31 documentos, escrita à mão para o projeto) está descrita em [docs/knowledge_base.md](docs/knowledge_base.md), incluindo os documentos parecidos de propósito e as lacunas intencionais usadas nos testes de recusa.

## Resultados da triagem

F1 macro no **teste** (2.082 tickets, split agrupado). O modelo é escolhido pela validação; o teste é avaliado uma vez. Detalhes, métricas por classe e comandos em [docs/experiments.md](docs/experiments.md).

| Modelo | Categoria (6 classes) | Urgência (3 classes) |
|---|---|---|
| TF-IDF (palavras + caracteres) + LogReg | **0,434** | **0,478** |
| Embeddings multilíngues + LogReg | pendente | pendente |

## API de triagem

Serviço FastAPI que carrega os campeões do Model Registry (ou de uma cópia exportada) no startup
([ADR 0005](docs/decisions/0005-servir-modelos-registry-ou-exportados.md)).

| Endpoint | Função |
|---|---|
| `POST /classify` | `{"text": "..."}` → categoria e urgência com confiança, versões dos modelos e latência |
| `GET /health` | Liveness + quais modelos (nome e versão) estão carregados |
| `GET /metrics` | Métricas Prometheus: requisições e latência por rota e status, predições por rótulo, versão dos modelos |
| `GET /docs` | OpenAPI interativo |

```bash
API_ALLOW_MODEL_DESERIALIZATION=true uv run uvicorn --factory supportai.api.main:create_app --port 8000

curl -s -X POST localhost:8000/classify -H 'content-type: application/json' \
  -d '{"text": "I was charged twice for my order, please refund the duplicate payment."}'
# {"request_id":"...","category":{"label":"billing_payments","confidence":0.9996},
#  "urgency":{"label":"high","confidence":0.891},"models":{...},"latency_ms":...}
```

- **Logs JSON** com `X-Request-ID`: o header recebido é propagado (ou um id é gerado), volta na resposta e aparece em todos os logs da requisição.
- **Entrada validada:** texto vazio, campos extras ou acima de `API_MAX_TEXT_CHARS` (5.000) retornam 422.
- **Falha cedo:** se algum modelo não carrega, o processo não sobe.

### Latência

`uv run python scripts/benchmark_latency.py --workers N`: 1.000 tickets reais do split de teste (404 caracteres em média), cliente HTTP na mesma máquina (4 vCPUs), após aquecimento. Resultados em `reports/benchmark_workers*.json`.

| Workers uvicorn | Concorrência | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| 1 | 1 | 9,1 ms | 13,6 ms | 16,6 ms | 102 req/s |
| 1 | 4 | 47,6 ms | 68,9 ms | 80,6 ms | 83 req/s |
| 4 | 4 | 12,1 ms | 28,5 ms | 36,5 ms | 261 req/s |

Com um processo, aumentar a concorrência **piora** o throughput: a inferência do TF-IDF é CPU-bound e segura o GIL, então as threads só disputam a CPU. Escalar por processos (workers ou réplicas) triplica o throughput. Cada worker ocupa ~380 MB de RSS, quase tudo de bibliotecas (`uv run python scripts/api_memory.py`).

### Docker

```bash
uv run python -m supportai.classifier.export          # campeões -> models/
docker build --target runtime-with-models -t supportai-api .
docker run -p 8000:8000 supportai-api

# ou, com os modelos montados do host:
docker compose --profile api up --build api
```

A imagem é multi-stage (uv no builder, só o venv no runtime), roda como usuário não-root, tem healthcheck em `/health` e instala apenas as dependências de serviço: `mlflow-skinny` no lugar do MLflow completo, sem pyarrow, matplotlib ou pandera. O CI constrói a imagem a cada push e reporta o tamanho: a separação das dependências levou o alvo `runtime` de **942 MB para 569 MB** (job "Imagem Docker" do CI, antes e depois do commit `build: separar dependências de serviço`).

### Deploy (Render ou Railway)

O registry e `models/` são locais (fora do git), então o deploy usa uma **imagem já construída**:

1. `uv run python -m supportai.classifier.export`
2. `docker build --target runtime-with-models -t ghcr.io/<usuario>/supportai-api:<versao> .` e `docker push` (GitHub Container Registry).
3. **Render:** New → Web Service → *Deploy an existing image* → a imagem acima. Health check path `/health`. A plataforma injeta `PORT`, que a imagem já usa.
   **Railway:** New → Deploy from Docker Image → a mesma imagem.
4. Plano: 1 worker cabe em 512 MB (~380 MB por processo). Para mais throughput, aumente as réplicas, não as threads.

> Estas instruções não foram executadas num deploy real a partir deste ambiente: a imagem é construída e testada no CI, mas o push para um registry e o deploy dependem das suas contas.

## Stack

Python 3.11+ · uv · FastAPI · Pydantic v2 · scikit-learn · sentence-transformers · MLflow · DVC · Pandera · PostgreSQL + pgvector · Langfuse/Phoenix · Evidently · Streamlit · pytest · ruff · mypy · GitHub Actions · Docker.

A camada de LLM é agnóstica de provedor (Anthropic, OpenAI ou Ollama local, escolhido por variável de ambiente).

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Docker.

```bash
# 1. Dependências (cria .venv a partir do uv.lock)
uv sync

# 2. Variáveis de ambiente
cp .env.example .env

# 3. Banco (Postgres 16 + pgvector)
docker compose up -d db

# 4. Qualidade
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest

# 5. Hooks de pre-commit (opcional, recomendado)
uv run pre-commit install
```

### Reproduzir a Fase 1

```bash
# Baixa o dataset (com checksum), limpa, valida, faz o split e treina o baseline.
# O DVC só refaz as etapas cujas dependências mudaram.
uv run dvc repro

# Promove o melhor run (pela validação) a "champion" no Model Registry
uv run python -m supportai.classifier.registry

# Interface do MLflow em http://localhost:5000
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- O remote do DVC é uma pasta local (`../supportai-dvc-storage`). Num clone novo, `uv run dvc repro` reconstrói tudo a partir do Hugging Face.
- Para carregar modelos do registry, o MLflow 3 exige `MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true` ([ADR 0004](docs/decisions/0004-mlflow-sqlite-registry-alias-champion.md)).
- O modelo de embeddings (`--model embeddings`) depende de torch CPU + sentence-transformers, que vão entrar como extra opcional (`uv sync --extra embeddings`) para não pesar no CI. *Pendente:* o extra ainda não está no `pyproject.toml`.

Testes que dependem do Postgres são marcados com `@pytest.mark.integration` e são pulados se o banco não estiver acessível.

## Estrutura

```
src/supportai/   código de produção (data, classifier, rag, llm, guardrails, agent, monitoring, api)
configs/         YAMLs de dados e treino (RAG e avaliação nas próximas fases)
data/            raw/processed (DVC) e knowledge_base/ (31 docs Markdown)
reports/         métricas versionadas (data.json, train_*.json)
scripts/         análises pontuais reproduzíveis
evals/           datasets e runner de avaliação
docs/            arquitetura, ADRs e experimentos
tests/           unit/ e integration/
app/             demo Streamlit
```

## Documentação

- [Arquitetura](docs/architecture.md)
- [Decisões (ADRs)](docs/decisions/)
- [Experimentos](docs/experiments.md)
