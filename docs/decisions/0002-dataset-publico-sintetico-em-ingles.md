# ADR 0002: dataset público sintético em inglês para a triagem

- **Status:** aceito
- **Data:** 2026-10-07

## Contexto

O plano original previa tickets em português. A busca não encontrou um dataset público de tickets
de suporte em português com rótulos de categoria **e** urgência e licença utilizável. As
alternativas eram:

1. gerar um dataset sintético em português com LLM;
2. usar um dataset público rotulado em outra língua.

Gerar com LLM custa dinheiro (contraria a regra de custo baixo), e o dataset final teria o viés do
nosso próprio prompt: o classificador aprenderia a "assinatura" do gerador, e nós mesmos teríamos
escolhido as categorias e a dificuldade.

## Decisão

Usar o **Tobi-Bueck/customer-support-tickets** (Hugging Face, **CC-BY-NC 4.0**), arquivo
`aa_dataset-tickets-multi-lang-5-2-50-version.csv`, revisão fixada, **somente o subconjunto em
inglês**. O produto inteiro (tickets e base de conhecimento) passa a ser em **inglês**; README e
documentação continuam em português.

- **Categoria** = `queue`, filtrada para as 6 filas mais frequentes (~89% dos tickets em inglês).
- **Urgência** = `priority` mapeada para 3 níveis (`very_low`→`low`, `critical`→`high`; a revisão
  usada já vem com `low/medium/high`).

## Justificativa

- Rótulos que não fomos nós que criamos: a avaliação é menos circular.
- Custo zero e reprodutível: revisão do repositório + SHA-256 do arquivo em `configs/data.yaml`.
- Tamanho suficiente (14.576 tickets após a limpeza) para comparar modelos com intervalos razoáveis.

## Trade-offs aceitos

- **Também é sintético.** O próprio autor descreve o dataset como gerado. Ele tem muitas
  paráfrases (tratadas no [ADR 0003](0003-split-agrupado-contra-quase-duplicatas.md)) e rótulos
  ruidosos: só 12% dos tickets da fila `returns_exchanges` mencionam "return", "exchange" ou
  "refund". Isso limita o teto de F1 de qualquer modelo, e as métricas absolutas não representam
  um e-commerce real.
- **Licença não comercial (NC).** Serve para um portfólio. Um uso comercial exigiria outro dataset.
- **Domínio desalinhado.** Os tickets falam muito de SaaS/TI corporativa, menos de e-commerce. A
  base de conhecimento foi escrita para ser coerente com as filas, não com cada ticket.
- **Perde-se o português.** Os modelos de embedding escolhidos são multilíngues, então a troca
  para português no futuro não exige mudar a arquitetura, só os dados.

## Quando revisitar

Se surgir um dataset real (ou anonimizado) de tickets em português, ou se o projeto for usado
comercialmente.
