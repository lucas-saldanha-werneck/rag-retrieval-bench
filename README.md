# rag-retrieval-bench

Demo independente, sem ligação com a cVortex. Não é um produto.

Compara três formas de recuperar trechos de uma base de FAQ, sem rede e sem dependências: busca por palavras (BM25), um índice vetorial de n-gramas de caracteres e a fusão dos dois (RRF). Mede hit@1, hit@3 e MRR e lista as perguntas em que cada método erra.

Eu fiz isto para ter algo concreto sobre a parte de RAG de uma vaga de engenheiro de IA generativa. Meus agentes em produção usam SQLite FTS5 (palavras-chave). Queria medir o que um índice vetorial muda antes de falar sobre isso.

## Os dados são inventados

`examples/faq_pt.jsonl` tem 24 trechos de FAQ de uma empresa fictícia de atendimento multicanal. `examples/queries_pt.jsonl` tem 28 perguntas, cada uma com o trecho certo marcado por mim. Nada vem de cliente real nem do site de ninguém.

## Como rodar

Python 3.10 ou mais novo. Não instala nada para rodar.

```sh
git clone https://github.com/lucas-saldanha-werneck/rag-retrieval-bench.git
cd rag-retrieval-bench
python -m ragbench
python -m ragbench --docs meus_trechos.jsonl --queries minhas_perguntas.jsonl --k 5
```

Formato: um JSON por linha. Trechos: `{"id": "d01", "text": "..."}`. Perguntas: `{"q": "...", "relevant": "d01"}`.

Testes: `pip install pytest && python -m pytest -q`.

## Resultado nos dados de exemplo

| método | hit@1 | hit@3 | MRR |
|---|---|---|---|
| bm25 | 0,64 | 0,71 | 0,72 |
| índice de n-gramas | 0,71 | 0,79 | 0,79 |
| híbrido (RRF) | 0,68 | 0,79 | 0,75 |

O que isso mostra:

- Nenhum método acerta as perguntas sem palavra em comum com o trecho. "O robô respondeu errado, como corrijo" deveria achar o trecho sobre o chatbot, mas o texto diz "chatbot". "O sistema está fora do ar?" deveria achar "instabilidade". O BM25 coloca esses trechos nas últimas posições (22 e 24 de 24). O índice de n-gramas melhora um deles (o do chatbot entra no top 3, por palavras parecidas) e deixa o outro em 10º, porque compara a forma das palavras, não o sentido.
- O índice de n-gramas ajuda com variação de palavra ("cancelamento" e "cancelar") e com erro de acento, e foi um pouco melhor que o BM25 aqui.
- A fusão não superou o melhor método isolado neste conjunto. Empatou em hit@3 e ficou abaixo em hit@1 e MRR.

Para fechar a lacuna de significado seria preciso um modelo de embeddings. Este repositório não tem um.

## O que foi testado

- 15 testes automáticos, offline, rodando no GitHub Actions: normalização de acentos, stopwords, n-gramas, BM25, índice de n-gramas, fusão, cálculo das métricas, rejeição de id inexistente, leitura da CLI.
- Um teste fixa os números da tabela acima (hit@3 de 20/28, 22/28 e 22/28). Se a tabela mudar, o teste falha.
- Os trechos e as perguntas de exemplo são consistentes entre si (ids únicos, todo `relevant` existe).

## O que não foi testado

- Embeddings neurais, bancos vetoriais (Qdrant, pgvector, Pinecone...) e reranqueadores. O "índice vetorial" aqui é TF-IDF de n-gramas em memória.
- Escala. São 24 trechos e 28 perguntas; uma diferença de 2 perguntas muda a tabela. Os números servem para comparar os métodos neste conjunto, não para concluir nada geral.
- Texto real de atendimento. As perguntas foram escritas por mim, que já conhecia os trechos, então podem favorecer algum método sem eu perceber.
- Outros idiomas e a lista de stopwords em português é curta.
- LangChain, LangGraph e geração de respostas: nada disso está aqui. É só a etapa de recuperação.

## Licença

MIT.
