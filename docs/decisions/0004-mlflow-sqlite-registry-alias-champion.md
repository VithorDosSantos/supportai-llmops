# ADR 0004: MLflow com SQLite, Model Registry com alias `champion` e serialização skops

- **Status:** aceito
- **Data:** 2026-10-07

## Contexto

Precisamos rastrear experimentos (params, métricas, artefatos), versionar modelos e dar à API
(Fase 2) uma forma estável de carregar "o modelo atual" sem mudar código a cada novo treino.

## Decisão

- **Backend SQLite** (`sqlite:///mlflow.db`) e artefatos em `mlartifacts/`, ambos locais e fora
  do git. O file store (`./mlruns`) não suporta o Model Registry e está obsoleto no MLflow 3.
- **Um modelo registrado por alvo:** `supportai-triage-category` e `supportai-triage-urgency`.
- **Alias `champion`** em vez de stages (`Production`/`Staging`, removidos no MLflow 3). A API
  carrega `models:/supportai-triage-<alvo>@champion`.
- **Seleção pelo F1 macro de validação**, nunca pelo de teste (`classifier/registry.py`). A
  promoção é idempotente: se o melhor run já é o campeão, nenhuma versão nova é criada.
- **Serialização skops** (padrão do MLflow 3), com os tipos próprios declarados explicitamente
  como confiáveis (`SKOPS_TRUSTED_TYPES`).
- O artefato é sempre um `Pipeline` que recebe texto cru: a API não precisa saber se o modelo é
  TF-IDF ou embeddings, e não há divergência entre o pré-processamento do treino e o do serviço.

## Justificativa

- Trocar o modelo em produção vira "mover o alias", com rollback trivial (apontar o alias para a
  versão anterior).
- SQLite não exige servidor: `mlflow ui --backend-store-uri sqlite:///mlflow.db` basta.
- skops não executa código arbitrário no carregamento, ao contrário do pickle.

## Trade-offs aceitos

- SQLite e artefatos locais não são compartilháveis entre máquinas. Em equipe, o próximo passo é
  um servidor MLflow com Postgres e artefatos em object storage (S3/GCS); só a URI muda.
- O MLflow 3.17 exige `MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true` para carregar qualquer modelo
  sklearn, inclusive skops, porque a lista de tipos confiáveis vem do próprio artefato. Aceitamos
  porque só carregamos modelos do nosso registry.
- O skops serializa cada escalar numpy como um arquivo separado. O vocabulário do TF-IDF
  (np.int64) deixava o salvamento em ~190 s; convertemos para `int` nativo após o `fit`
  (`compact_tfidf_vocabulary`), o que reduziu para < 1 s com predições idênticas.

## Quando revisitar

Ao fazer deploy (Fase 2), caso a API rode em outra máquina: aí o registry precisa ser remoto.
