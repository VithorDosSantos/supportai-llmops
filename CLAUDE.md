# CLAUDE.md — SupportAI: Plataforma Inteligente de Atendimento ao Cliente

> Este arquivo é o guia do projeto para o Claude Code. Leia-o inteiro antes de qualquer tarefa e siga-o em todas as sessões.

## 1. Contexto e objetivo

Sou um desenvolvedor construindo um **projeto de portfólio para vagas de ML Engineer e AI Engineer**. O objetivo não é só "funcionar": é demonstrar, com código de qualidade de produção e métricas reais, que sei resolver os problemas que as empresas mais enfrentam nessas áreas:

| Dor real nas empresas | Como o projeto demonstra a solução |
|---|---|
| Modelo funciona no notebook, mas não vai para produção | API containerizada, CI/CD, deploy |
| Falta de reprodutibilidade | MLflow + DVC + configs versionadas + seeds fixas |
| Modelo degrada com o tempo (drift) | Monitoramento de drift com relatórios |
| Dados ruins quebram pipelines | Validação de schema e qualidade de dados |
| RAG alucina e ninguém mede | Suíte de avaliação automatizada com métricas |
| Mudanças de prompt sem saber se melhorou | Avaliação de regressão rodando no CI |
| Custo e latência altos de LLM | Cache semântico, roteamento de modelos, métricas de custo |
| Segurança e LGPD | Mascaramento de PII, defesa contra prompt injection |
| Sistema caixa-preta | Tracing e observabilidade de ponta a ponta |
| Agentes imprevisíveis | Tool calling com aprovação humana e avaliação de tarefas |

**Importante:** eu quero APRENDER com este projeto. Ao implementar algo não trivial, explique brevemente o porquê das decisões e os trade-offs (em comentários curtos ou na resposta), para que eu consiga defender cada escolha em entrevistas.

## 2. O produto

Sistema que recebe tickets de suporte de um e-commerce fictício (a loja **Voltix**) e:

> Idioma: tickets e base de conhecimento em **inglês** (não há dataset público adequado em português; ver `docs/decisions/0002-*`). README e docs em português.

1. **Triagem (ML clássico):** classifica a categoria (ex.: entrega, pagamento, reembolso, produto com defeito, conta) e a urgência do ticket.
2. **Resposta sugerida (RAG):** busca na base de conhecimento (FAQs, políticas de troca/reembolso, manuais) e gera uma resposta citando as fontes. Se não houver informação suficiente, recusa em vez de inventar.
3. **Agente (opcional, fase 6):** usa ferramentas para consultar status de pedido e solicitar reembolso numa API simulada, exigindo aprovação humana para ações sensíveis.

## 3. Stack

- **Linguagem:** Python 3.11+
- **Gerenciamento de dependências:** `uv` (com `pyproject.toml`)
- **API:** FastAPI + Pydantic v2
- **ML clássico:** scikit-learn, sentence-transformers
- **Experimentos e registro de modelos:** MLflow
- **Versionamento de dados:** DVC
- **Validação de dados:** Pandera
- **Banco / busca vetorial:** PostgreSQL + pgvector (via Docker Compose)
- **Busca lexical:** BM25 (rank-bm25 ou full-text search do Postgres)
- **Reranking:** cross-encoder do sentence-transformers
- **LLM:** camada de abstração com provedor configurável por variável de ambiente (Anthropic, OpenAI ou Ollama local). Nunca acoplar o código a um único provedor.
- **Avaliação de LLM:** RAGAS e/ou LLM-as-judge próprio
- **Observabilidade:** Langfuse (self-hosted via Docker) ou Arize Phoenix
- **Drift:** Evidently
- **Interface de demo:** Streamlit
- **Qualidade:** pytest, ruff, mypy, pre-commit
- **CI/CD:** GitHub Actions
- **Containers:** Docker + Docker Compose

Se alguma biblioteca estiver desatualizada ou houver alternativa claramente melhor, proponha antes de trocar.

## 4. Estrutura do repositório

```
supportai/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .github/workflows/        # ci.yml (lint, testes), eval.yml (avaliação de regressão)
├── configs/                  # YAMLs de treino, RAG e avaliação
├── data/
│   ├── raw/                  # versionado com DVC
│   ├── processed/
│   └── knowledge_base/       # documentos da base de conhecimento
├── docs/
│   ├── architecture.md       # diagrama (Mermaid) e decisões
│   ├── decisions/            # ADRs: um arquivo por decisão importante
│   └── experiments.md        # tabelas de resultados de cada experimento
├── notebooks/                # apenas exploração; nada de lógica de produção aqui
├── src/supportai/
│   ├── data/                 # ingestão, validação, schemas
│   ├── classifier/           # treino, avaliação, inferência da triagem
│   ├── rag/                  # chunking, indexação, retrieval, reranking, geração
│   ├── llm/                  # abstração de provedores, cache, roteamento
│   ├── guardrails/           # PII, prompt injection
│   ├── agent/                # ferramentas e loop do agente (fase 6)
│   ├── monitoring/           # drift, métricas de custo/latência
│   ├── api/                  # FastAPI: rotas, dependências
│   └── config.py             # settings via pydantic-settings
├── evals/
│   ├── datasets/             # conjuntos de teste (perguntas + respostas esperadas)
│   └── run_eval.py
├── app/                      # interface Streamlit
└── tests/                    # unit/ e integration/
```

## 5. Roadmap por fases

Trabalhe **uma fase por vez**. Ao final de cada fase: rode lint e testes, atualize o README e o `docs/experiments.md`, faça um resumo do que foi feito e **pergunte antes de iniciar a próxima fase**.

### Fase 0 — Setup
- [x] Estrutura de pastas, `pyproject.toml` com uv, ruff, mypy, pytest, pre-commit
- [x] `docker-compose.yml` com Postgres + pgvector
- [x] `config.py` com pydantic-settings e `.env.example`
- [x] Workflow de CI básico (lint + testes)
- [x] README inicial com objetivo e como rodar

### Fase 1 — Dados e classificador de triagem
- [x] Obter dados: procurar dataset público de tickets de suporte em português (Hugging Face/Kaggle). Se não houver um adequado, gerar dataset sintético com LLM e **documentar isso honestamente** no README, incluindo limitações.
- [x] Gerar também a base de conhecimento (FAQs e políticas do e-commerce fictício, ~20–40 documentos em Markdown)
- [x] Schema e validação com Pandera; pipeline de limpeza
- [x] Versionar dados com DVC
- [x] Baseline: TF-IDF + regressão logística
- [ ] Modelo 2: embeddings (sentence-transformers multilíngue) + classificador *(código e testes prontos; falta a execução real)*
- [x] Split estratificado, métricas: F1 macro, matriz de confusão, relatório por classe
- [x] Tudo rastreado no MLflow; melhor modelo registrado no Model Registry
- [ ] Tabela comparativa em `docs/experiments.md`

### Fase 2 — Servir em produção
- [ ] API FastAPI: `POST /classify`, `GET /health`, `GET /metrics`
- [ ] Carregar modelo do MLflow Registry na inicialização
- [ ] Dockerfile multi-stage enxuto
- [ ] Testes unitários e de integração da API
- [ ] Logging estruturado (JSON) com request id
- [ ] Script de benchmark de latência (p50/p95/p99) com resultados no README
- [ ] Instruções de deploy (Render/Railway ou cloud free tier)

### Fase 3 — RAG
- [ ] Ingestão da base de conhecimento com metadados (fonte, seção)
- [ ] Comparar 2–3 estratégias de chunking (tamanho fixo, por seção/Markdown, com overlap)
- [ ] Indexação no pgvector
- [ ] Busca híbrida (BM25 + vetorial com Reciprocal Rank Fusion) + reranking com cross-encoder
- [ ] Geração com citação das fontes em formato estruturado
- [ ] Recusa explícita quando o contexto não sustenta a resposta
- [ ] Endpoint `POST /answer` que combina triagem + RAG

### Fase 4 — Avaliação (diferencial principal do projeto)
- [ ] Dataset de avaliação com 100–200 exemplos: pergunta, documentos relevantes esperados, resposta de referência. Incluir perguntas **sem resposta** na base, para testar a recusa.
- [ ] Métricas de retrieval: recall@k, MRR, nDCG
- [ ] Métricas de geração: fidelidade às fontes (faithfulness), relevância da resposta, taxa de recusa correta
- [ ] Se usar LLM-as-judge: validar o juiz contra ~30 rótulos manuais e reportar a concordância
- [ ] Experimentos documentados: chunking A vs. B vs. C; com e sem reranker; vetorial vs. híbrida
- [ ] Workflow `eval.yml` no CI: roda a avaliação e **falha se as métricas caírem** abaixo de um limiar definido em config

### Fase 5 — Custo, segurança e observabilidade
- [ ] Contabilizar tokens, custo e latência por requisição
- [ ] Cache semântico (respostas a perguntas similares)
- [ ] Roteamento de modelos: perguntas simples → modelo barato; complexas → modelo maior. Medir o impacto em custo e qualidade (rodando a suíte de avaliação).
- [ ] Guardrails: detectar e mascarar PII (CPF, e-mail, telefone, cartão) antes de enviar ao LLM e nos logs
- [ ] Defesa contra prompt injection, com casos de teste adversariais na suíte de avaliação
- [ ] Tracing de ponta a ponta (Langfuse ou Phoenix)
- [ ] Monitoramento de drift do classificador com Evidently: simular dados novos com distribuição alterada e gerar relatório
- [ ] Tabela "antes vs. depois" de custo e latência no README

### Fase 6 (opcional) — Agente
- [ ] API simulada de pedidos (status, reembolso)
- [ ] Agente com tool calling; ações sensíveis (reembolso) exigem aprovação humana (human-in-the-loop)
- [ ] Limites: número máximo de passos, timeouts, tratamento de erro das ferramentas
- [ ] Avaliação do agente: taxa de sucesso por tarefa, uso incorreto de ferramentas

### Fase 7 — Apresentação
- [ ] Interface Streamlit: enviar ticket, ver categoria, resposta, fontes citadas, custo e latência
- [ ] README final como um case de negócio: problema, arquitetura (Mermaid), decisões e trade-offs, resultados com números, limitações e próximos passos
- [ ] ADRs para as decisões principais em `docs/decisions/`
- [ ] Seção "O que não funcionou e por quê"

## 6. Padrões de código

- Type hints em todo código de `src/`; mypy sem erros
- Funções pequenas, responsabilidades claras; nada de lógica de produção em notebooks
- Configurações em YAML/variáveis de ambiente, nunca hardcoded
- **Nunca** commitar segredos; usar `.env` (no `.gitignore`) e `.env.example`
- Seeds fixas para reprodutibilidade
- Toda funcionalidade nova vem com testes; mocks para chamadas de LLM nos testes unitários (testes não devem gastar dinheiro com API)
- Commits pequenos e com mensagens no padrão Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`...)
- Docstrings curtas explicando o "porquê", não o óbvio

## 7. Regras de trabalho para o Claude Code

1. Antes de começar uma fase, apresente um plano curto e espere minha aprovação.
2. Rode `ruff`, `mypy` e `pytest` antes de declarar uma tarefa concluída.
3. **Nunca invente métricas.** Todo número no README e em `docs/experiments.md` deve vir de uma execução real, com o comando para reproduzi-la.
4. Se uma decisão tiver trade-offs relevantes (ex.: pgvector vs. Qdrant, tamanho de chunk), registre um ADR curto.
5. Mantenha o custo baixo: prefira modelos baratos ou locais durante o desenvolvimento e avise antes de rodar algo que faça muitas chamadas pagas.
6. Se algo do plano não fizer sentido técnico, diga e proponha alternativa.
7. Ao fim de cada fase, sugira 3–5 perguntas de entrevista que eu deveria saber responder sobre o que foi construído.
