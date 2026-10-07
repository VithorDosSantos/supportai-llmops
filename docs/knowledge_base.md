# Base de conhecimento

31 documentos em Markdown (inglês) em `data/knowledge_base/`, da **Voltix**, uma loja online
fictícia de eletrônicos que também vende assinaturas (Voltix Care+, Voltix Cloud Backup) e um
produto B2B (Voltix Business). O domínio usado é `voltix.example`, reservado para exemplos, para
não imitar nenhuma empresa real. Os documentos foram escritos à mão para este projeto.

## Formato

```markdown
---
id: kb-refund-policy          # estável: é o que o RAG cita
title: Refund Policy          # igual ao H1
category: returns_exchanges   # uma das 6 categorias da triagem
---

# Refund Policy
## Seções...
```

`tests/unit/test_knowledge_base.py` valida a base real: front-matter, ids únicos, categoria
existente na triagem, H1 igual ao título e **referências cruzadas** (`see "Título"`) apontando
para documentos que existem. Esse teste já encontrou uma referência quebrada na primeira versão.

## Distribuição por categoria

| Categoria | Docs |
|---|---|
| returns_exchanges | 6 |
| billing_payments | 6 |
| customer_service | 5 |
| technical_support | 5 |
| product_support | 5 |
| it_support | 4 |

As categorias são as mesmas filas do classificador. Na Fase 3, isso permite usar a categoria
prevista como filtro ou sinal de reranking no retrieval (e medir se ajuda).

## Documentos parecidos de propósito

Para que o retrieval seja difícil de verdade e a avaliação (Fase 4) consiga distinguir
estratégias, há grupos de documentos com vocabulário muito próximo e respostas diferentes:

| Grupo | Documentos | Armadilha |
|---|---|---|
| Devolução x troca x reembolso | return-policy, exchange-policy, refund-policy, return-shipping-labels | Mesmo prazo (30 dias, 45 com Care+), mas taxas diferentes: taxa de reposição de 15% vale para devolução e não para troca da mesma variante |
| Avaria na chegada x garantia | damaged-or-wrong-item, warranty-claim-process, warranty-coverage | Até 48 h após a entrega é "avaria na chegada" (troca expressa); depois disso é garantia |
| Cobrança duplicada x pagamento recusado | double-charges-and-pending-authorizations, failed-payments | Pré-autorização (some sozinha em 3 a 5 dias úteis) x cobrança real |
| Assinaturas | subscription-plans-and-billing, cancelling-a-subscription, failed-payments | Upgrade cobra proporcional; plano anual só é reembolsado em até 14 dias |
| Alterar x cancelar pedido | change-or-cancel-order, order-tracking | Edição só no status Received; Preparing não permite editar |
| Dados de backup | cloud-backup-restore, cancelling-a-subscription, failed-payments | 30 dias de retenção, com significados diferentes em cada contexto |

## Lacunas intencionais (perguntas sem resposta na base)

A base **não** cobre os temas abaixo. Eles vão compor as perguntas "sem resposta" do dataset de
avaliação (Fase 4), para medir se o RAG recusa em vez de inventar. Ao editar a base, não cubra
esses temas sem atualizar o dataset de avaliação.

- Envio internacional, alfândega e impostos de importação (a base só fala de prazos e preços de envio).
- Cartões-presente, crédito na loja e cupons de desconto.
- Cobertura de preço (price matching) e proteção contra queda de preço.
- Programa de troca de aparelho usado (trade-in) e descontos para estudantes.
- Parcelamento e "compre agora, pague depois" (a base lista os meios de pagamento, mas não fala de parcelas).
- Reparo presencial ou visita técnica.
- Recuperação de dados de discos fisicamente danificados.
- Instalação on-premise do Voltix Business e certificações de conformidade (HIPAA, SOC 2).
- Versionamento ou depreciação da API.

Algumas perguntas ficam **perto** de conteúdo existente (ex.: "posso pagar em 12 parcelas?" perto
de *Accepted Payment Methods*). São esses os casos em que um RAG costuma alucinar.
