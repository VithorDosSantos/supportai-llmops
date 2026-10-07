# Arquitetura

> Documento vivo: cada fase atualiza o diagrama e as decisões. Componentes ainda
> não implementados aparecem tracejados.

## Visão geral (alvo)

```mermaid
flowchart LR
    U[Cliente / Atendente] -->|ticket| API[FastAPI]

    subgraph Triagem [Triagem - ML clássico]
        CLS[Classificador<br/>categoria + urgência]
    end

    subgraph RAG [Resposta sugerida - RAG]
        GR[Guardrails<br/>PII + prompt injection]
        RET[Busca híbrida<br/>BM25 + vetorial + RRF]
        RR[Reranker<br/>cross-encoder]
        GEN[Geração com citações<br/>ou recusa]
    end

    API --> CLS
    API --> GR --> RET --> RR --> GEN
    RET <--> PG[(Postgres + pgvector)]
    GEN <--> LLM[Camada LLM<br/>Anthropic / OpenAI / Ollama]

    MLF[(MLflow Registry)] -. modelo .-> CLS
    API -. traces .-> OBS[Langfuse / Phoenix]
    CLS -. dados de produção .-> DRIFT[Evidently]
    EVAL[Suíte de avaliação] -. CI .-> GEN

    classDef todo stroke-dasharray: 5 5
    class CLS,GR,RET,RR,GEN,LLM,MLF,OBS,DRIFT,EVAL,API todo
```

## Estado atual (Fase 0)

- Pacote `supportai` com layout `src/`, um subpacote por responsabilidade.
- Configuração central em `src/supportai/config.py` (pydantic-settings), lida de variáveis de ambiente / `.env`.
- Postgres 16 + pgvector via `docker compose`.
- CI: ruff, mypy (strict), pytest, com Postgres + pgvector como service container.

## Estado atual (Fase 1)

Pipeline de dados e treino da triagem, orquestrado pelo DVC (`dvc.yaml`):

```mermaid
flowchart LR
    HF[(Hugging Face<br/>revisão fixada)] -->|download<br/>+ SHA-256| RAW[data/raw/tickets.csv]
    RAW -->|prepare| CLEAN[Limpeza + Pandera]
    CLEAN --> GROUP[Grupos de quase-duplicatas<br/>TF-IDF char, cosseno ≥ 0,8]
    GROUP --> SPLIT[StratifiedGroupKFold<br/>train / val / test]
    SPLIT --> PQ[(data/processed/*.parquet)]
    PQ -->|train_tfidf| TF[TF-IDF + LogReg]
    PQ -.->|train embeddings| EMB[Encoder multilíngue + LogReg]
    TF --> MLF[(MLflow<br/>SQLite + artefatos)]
    EMB -.-> MLF
    MLF -->|registry.py:<br/>melhor val_f1_macro| REG[Model Registry<br/>alias champion]
```

- `src/supportai/data/`: config tipada (`configs/data.yaml`), download, schemas Pandera, limpeza,
  split agrupado e loader da base de conhecimento.
- `src/supportai/classifier/`: pipelines sklearn (texto cru → rótulo), métricas, treino com MLflow
  e promoção do campeão.
- Métricas versionadas no git em `reports/` (`data.json`, `train_*.json`).

## Decisões

Ver [ADRs](decisions/).
