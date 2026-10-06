# SupportAI: atendimento ao cliente com ML, RAG avaliado e LLMOps

[![CI](https://github.com/VithorDosSantos/supportai-llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/VithorDosSantos/supportai-llmops/actions/workflows/ci.yml)

Sistema que recebe tickets de suporte de um e-commerce fictício (em português) e:

1. **Triagem (ML clássico):** classifica categoria (entrega, pagamento, reembolso, produto com defeito, conta) e urgência.
2. **Resposta sugerida (RAG):** busca na base de conhecimento e gera uma resposta **citando as fontes**, ou recusa quando não há informação suficiente.
3. **Agente (opcional):** consulta pedidos e solicita reembolsos numa API simulada, com aprovação humana para ações sensíveis.

O foco não é só "funcionar". O projeto ataca problemas comuns de levar ML/LLM para produção: reprodutibilidade, avaliação automatizada de RAG, regressão de prompts no CI, custo e latência, PII/LGPD, observabilidade e drift.

> **Status:** Fase 0 (setup) concluída. Os números de desempenho vão entrar no README conforme as fases forem executadas. Todo número publicado aqui vem de uma execução real, com o comando para reproduzi-lo.

## Roadmap

| Fase | Conteúdo | Status |
|---|---|---|
| 0 | Setup: estrutura, tooling, Postgres + pgvector, config, CI | ✅ |
| 1 | Dados + classificador de triagem (TF-IDF vs. embeddings, MLflow, DVC) | ⏳ |
| 2 | Servir em produção: FastAPI, Docker, benchmark de latência | ⏳ |
| 3 | RAG: chunking, busca híbrida, reranking, citações, recusa | ⏳ |
| 4 | Avaliação: retrieval + geração, regressão no CI | ⏳ |
| 5 | Custo, cache semântico, roteamento, PII, prompt injection, tracing, drift | ⏳ |
| 6 | Agente com tool calling e human-in-the-loop (opcional) | ⏳ |
| 7 | Demo Streamlit, ADRs, case de negócio | ⏳ |

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

Testes que dependem do Postgres são marcados com `@pytest.mark.integration` e são pulados se o banco não estiver acessível.

## Estrutura

```
src/supportai/   código de produção (data, classifier, rag, llm, guardrails, agent, monitoring, api)
configs/         YAMLs de treino, RAG e avaliação
data/            raw/processed (DVC) e knowledge_base/
evals/           datasets e runner de avaliação
docs/            arquitetura, ADRs e experimentos
tests/           unit/ e integration/
app/             demo Streamlit
```

## Documentação

- [Arquitetura](docs/architecture.md)
- [Decisões (ADRs)](docs/decisions/)
- [Experimentos](docs/experiments.md)
