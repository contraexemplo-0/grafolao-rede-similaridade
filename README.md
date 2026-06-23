# Rede de Similaridade de Palpites  Grafolão da Copa

## 1. Visão geral

Este projeto constrói e analisa uma rede de similaridade de palpites do Grafolão da Copa.

A ideia central é representar participantes e relações de proximidade entre seus palpites:

```txt
vértices = participantes
arestas = relações de similaridade entre participantes
peso da aresta = intensidade da similaridade
```

O projeto foi desenvolvido como um pipeline acadêmico em Python, com foco em modelagem, reprodutibilidade, testes automatizados e geração de tabelas/figuras para análise.

## 2. Pergunta investigada

A pergunta principal é:

```txt
Quem pensa como quem no bolão?
```

Em vez de medir quem acertou mais resultados, o projeto mede quão próximos são os padrões de palpite entre participantes. Assim, participantes conectados por arestas fortes tendem a fazer palpites semelhantes ao longo dos jogos considerados.

## 3. Escopo do projeto

Este projeto é um módulo analítico offline. Ele não é:

- aplicação web;
- API;
- dashboard interativo;
- ranking tradicional de acertos;
- ranking de originalidade;
- prova de cópia;
- prova de influência causal.

O ranking produzido pelo projeto é um ranking de centralidade estrutural na rede de similaridade, não um ranking esportivo tradicional.

## 4. Formato dos dados de entrada

O pipeline principal lê quatro CSVs canônicos:

```txt
participantes.csv
jogos.csv
palpites.csv
resultados.csv
```

Papel de cada arquivo:

- `participantes.csv`: lista de participantes anonimizados.
- `jogos.csv`: lista de jogos considerados no recorte.
- `palpites.csv`: placares previstos pelos participantes para cada jogo.
- `resultados.csv`: resultados reais dos jogos.

Colunas mínimas esperadas:

`participantes.csv`

```txt
participante_id
rotulo
```

`jogos.csv`

```txt
jogo_id
num
rodada
fase
grupo
time_a
time_b
data_hora
status
```

`palpites.csv`

```txt
participante_id
jogo_id
gols_a_palpite
gols_b_palpite
```

Colunas opcionais aceitas em `palpites.csv`, quando disponíveis:

```txt
status_palpite
pontos
```

`resultados.csv`

```txt
jogo_id
gols_a_real
gols_b_real
status
```

A pasta `data/example/` contém uma base fictícia e segura nesse formato. Ela é a forma recomendada para validar rapidamente o funcionamento do projeto em uma base pequena e compreensível.

A pasta `data/processed/` contém a base real anonimizada usada no experimento principal. Ela também segue o mesmo contrato dos quatro CSVs canônicos.

Organização dos dados:

```txt
data/example/   = base fictícia segura
data/processed/ = base real anonimizada segura
data/raw/       = dados brutos privados
data/private/   = dados privados auxiliares
*.sql           = dumps privados
```

Para usar o projeto com novos dados, forneça arquivos no mesmo formato canônico. O fluxo público do projeto parte de `data/example/`, de `data/processed/` ou de CSVs anonimizados próprios.

## 5. Modelagem da similaridade

### 5.1 Similaridade entre dois palpites

Dois participantes são comparados jogo a jogo. Um palpite é um placar previsto, por exemplo:

```txt
2 x 1
0 x 0
1 x 3
```

A comparação entre dois palpites considera:

- resultado previsto: vitória do time A, empate ou vitória do time B;
- proximidade do saldo de gols;
- proximidade do placar.

Intuição:

- palpites idênticos recebem similaridade máxima;
- palpites com mesmo vencedor, mas placar diferente, ainda são parecidos;
- palpites com resultados opostos são pouco parecidos;
- placares muito distantes são penalizados.

As medidas `prox_saldo` e `prox_placar` são normalizadas entre 0 e 1.

Se os palpites indicam o mesmo resultado:

```txt
sim_jogo = 0.50 + 0.25 * prox_saldo + 0.25 * prox_placar
```

Se os palpites indicam resultados diferentes:

```txt
sim_jogo = 0.25 * prox_placar
```

### 5.2 Similaridade entre dois participantes

Para dois participantes `u` e `v`, o projeto considera apenas os jogos em que ambos possuem palpite válido.

A similaridade média é:

```txt
sim_media = média das similaridades jogo a jogo nos jogos comparáveis
```

Depois, essa média é ajustada pela cobertura.

### 5.3 Cobertura

Cobertura mede a proporção de jogos comparáveis:

```txt
cobertura = número de jogos em que ambos palpitaram / número total de jogos considerados
```

Isso evita concluir que dois participantes são muito parecidos com base em poucos jogos.

No modelo oficial:

```txt
sim_participantes = sim_media * cobertura
```

### 5.4 Modelo oficial e modelo suavizado

Modelo oficial:

```txt
sim_participantes = sim_media * cobertura
```

Experimento complementar com cobertura suavizada:

```txt
sim_participantes = sim_media * cobertura^0.5
```

O modelo suavizado é uma análise de sensibilidade. Ele não substitui automaticamente o modelo oficial. Seu objetivo é avaliar como a penalização por cobertura afeta a topologia da rede.

## 6. Construção do grafo

O grafo de similaridade é:

- não direcionado;
- simples;
- ponderado;
- filtrado por limiar.

Cada participante é um vértice. Uma aresta entre `u` e `v` existe quando:

```txt
sim_participantes(u, v) >= theta
```

Cada aresta armazena:

```txt
weight = similaridade
distance = 1 - weight
```

Uso de `weight`:

- PageRank;
- força dos vértices;
- espessura visual de arestas;
- interpretação de proximidade/similaridade.

Uso de `distance`:

- caminhos mínimos;
- betweenness;
- distância média;
- diâmetro.

## 7. Algoritmos e métricas utilizados

### Componentes conexas

Usadas para identificar partes desconectadas da rede, incluindo participantes isolados e o tamanho da maior componente.

### Comunidades por modularidade gulosa

As comunidades são detectadas com o algoritmo de modularidade gulosa do NetworkX:

```python
networkx.algorithms.community.greedy_modularity_communities
```

Esse algoritmo identifica subgrupos mais densamente conectados dentro da rede, usando `weight` como peso das arestas.

### PageRank ponderado

O PageRank ponderado gera o Ranking de Centralidade na Rede de Similaridade.

Importante:

```txt
PageRank não mede quem acertou mais palpites.
PageRank mede quem está mais central na rede de similaridade.
```

### Betweenness centrality

A betweenness ajuda a identificar participantes que atuam como ponte entre partes da rede. Essa métrica usa `distance = 1 - weight`.

### Cliques maximais

Cliques maximais identificam subconjuntos em que todos os participantes estão conectados entre si. Essa análise é complementar.

### Distâncias, diâmetro e distância média

Distâncias, diâmetro e distância média são calculados com `distance`, não com `weight`, porque valores maiores de similaridade representam menor distância relacional.

### Metagrafo de comunidades

O metagrafo é uma visualização agregada:

```txt
nó = comunidade
tamanho do nó = quantidade de participantes
aresta = relação agregada entre comunidades
```

Ele é mais legível que o grafo completo quando a rede possui muitos vértices e arestas.

## 8. Pipeline de execução

Fluxo geral:

```txt
CSVs de entrada
        ↓
cálculo da similaridade entre palpites
        ↓
cálculo da similaridade entre participantes
        ↓
matriz/tabela de pares
        ↓
construção do grafo
        ↓
métricas globais e individuais
        ↓
componentes, comunidades, cliques e ranking
        ↓
figuras e relatórios
        ↓
exportação GraphML/JSON
```

## 9. Como executar com dados de exemplo e dados reais anonimizados

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 9.1 Execução didática com dados fictícios

Execute o pipeline com a base fictícia:

```bash
python src/main.py --data-dir data/example --output-dir outputs/example --theta 0.5
```

Essa execução didática usa `data/example/`, uma base fictícia segura. Ela serve para validar rapidamente o pipeline em uma base pequena e compreensível.

### 9.2 Execução do experimento real anonimizado

Para reproduzir o experimento real anonimizado, use `data/processed/`:

```bash
python src/main.py --data-dir data/processed --output-dir outputs/final_theta_0_20 --theta 0.2
```

Essa execução usa a base real já anonimizada e segura para reproduzir os resultados principais do trabalho.

## 10. Como usar com novos dados

Para usar o projeto com novos dados, crie uma pasta com os quatro CSVs canônicos:

```txt
participantes.csv
jogos.csv
palpites.csv
resultados.csv
```

Use `data/example/` como referência de formato.

Exemplo:

```bash
python src/main.py --data-dir caminho/para/csvs --output-dir outputs/meu_experimento --theta 0.5
```

Na prática, o usuário pode trocar:

- `data/processed` pela pasta onde estão seus próprios CSVs canônicos;
- `outputs/final_theta_0_20` por uma pasta própria de saída.

Para reproduzir os resultados principais deste projeto, use `data/processed/`, que contém dados reais já anonimizados. Para aplicar o pipeline a outro contexto, forneça CSVs anonimizados no mesmo formato canônico.

Comandos complementares para quem já possui CSVs canônicos anonimizados:

```bash
python scripts/analisar_limiar.py --data-dir data/processed --output-dir outputs --thetas 0.2,0.3,0.4,0.5,0.6
python scripts/gerar_visualizacoes.py --output-dir outputs/final_theta_0_20 --theta 0.2
python scripts/exportar_resultados_finais.py --output-dir outputs/final_theta_0_20 --theta 0.2
python scripts/experimento_cobertura.py --data-dir data/processed --output-dir outputs/experimentos/cobertura_suavizada_gamma_0_5 --theta 0.2 --gamma 0.5
```

Nota: `data/processed/` foi obtido previamente a partir de dados reais anonimizados. O processo de obtenção a partir de dados brutos privados não é o fluxo público principal do README.

## 11. Saídas geradas

Cada execução do pipeline organiza resultados em subpastas:

```txt
tabelas/
figuras/
grafos/
resumo/
```

Arquivos importantes:

- `matriz_similaridade_final.csv`: matriz quadrada de similaridade entre participantes.
- `pares_similaridade_final.csv`: tabela de similaridade por par de participantes.
- `resumo_grafo_final.csv`: métricas globais do grafo.
- `metricas_participantes_final.csv`: métricas individuais por participante.
- `ranking_centralidade.csv`: ranking por PageRank ponderado.
- `resumo_comunidades_maior_componente.csv`: resumo das comunidades na maior componente.
- `vertices_ponte_intercomunidades.csv`: participantes que conectam comunidades.
- `grafo_final.graphml`: grafo exportado em GraphML.
- `grafo_final.json`: grafo exportado em JSON.
- `relatorio_final.md`: relatório consolidado.

Alguns artefatos, como `grafo_final.graphml`, `grafo_final.json` e `relatorio_final.md`, são gerados pelo script de exportação final:

```bash
python scripts/exportar_resultados_finais.py --output-dir outputs/final_theta_0_20 --theta 0.2
```

Figuras importantes:

- `distribuicao_tamanho_componentes_melhorado.png`;
- `heatmap_similaridade_ordenado.png`;
- `ranking_centralidade_pagerank_melhorado.png`;
- `metagrafo_comunidades.png`;
- `grafo_pontes_intercomunidades.png`;
- `comparacao_cobertura.png`.

Os grafos de visualização podem usar filtros para reduzir poluição visual. As métricas são calculadas no grafo completo.

## 12. Dashboard local: módulo Redes de Similaridade

Além do pipeline acadêmico, o projeto possui um protótipo local em Streamlit para demonstrar como o módulo **Redes de Similaridade** poderia aparecer dentro do produto Grafolão.

Esse dashboard é uma demonstração visual e local. Ele não substitui o artigo, não altera a modelagem, não integra com banco de dados, não possui autenticação e não se conecta ao sistema real do Grafolão.

Funcionalidades demonstradas:

- **Visão geral da rede**: resumo do modelo selecionado.
- **Meus semelhantes**: participantes com padrões de palpite próximos ao usuário demo.
- **Grupos de palpite**: comunidade do usuário demo e membros relacionados.
- **Participantes centrais**: ranking estrutural por centralidade PageRank.
- **Mapa da rede**: visualização interativa da rede de similaridade.

O dashboard usa um participante artificial chamado `P_TESTE`. Esse usuário foi criado apenas para demonstrar uma experiência personalizada, permitindo visualizar seus participantes mais semelhantes, sua comunidade, sua posição no ranking de centralidade e o mapa da rede.

Os dados usados pelo dashboard são anonimizados. A base demo fica em:

```txt
dashboard/data/demo/
```

Os resultados consumidos pelo dashboard ficam em:

```txt
outputs/dashboard_demo/linear/
outputs/dashboard_demo/suavizado/
```

Para instalar as dependências específicas do dashboard:

```bash
pip install -r dashboard/requirements.txt
```

Se necessário, gere a base demo com `P_TESTE`:

```bash
python dashboard/gerar_dados_demo.py
```

Em seguida, gere os resultados usados pela interface:

```bash
python dashboard/gerar_resultados_demo.py
```

Para rodar o dashboard:

```bash
streamlit run dashboard/app.py
```

O dashboard não usa dump SQL privado, não acessa `data/raw/` ou `data/private/`, não expõe dados pessoais e não deve ser tratado como fluxo público de extração de dados. Seu objetivo é demonstrar como os resultados da modelagem em grafos poderiam ser apresentados a um usuário final.

## 13. Resultados obtidos no experimento real anonimizado

Os números abaixo vêm de um experimento real anonimizado disponível em `data/processed/`.

### Modelo oficial

```txt
theta = 0.20
gamma = 1.0
participantes válidos = 135
jogos válidos = 33
palpites válidos = 1113
vértices = 135
arestas = 479
componentes conexas = 93
componentes unitárias = 92
maior componente = 43
participante mais central = P126
```

### Modelo com cobertura suavizada

```txt
theta = 0.20
gamma = 0.5
vértices = 135
arestas = 1881
componentes conexas = 27
componentes unitárias = 26
maior componente = 109
participante mais central = P015
```

O modelo suavizado tornou a rede mais conectada, indicando que a estrutura observada depende da forma de penalização da cobertura.

## 14. Interpretação dos resultados

A rede oficial com cobertura linear ficou bastante fragmentada. Isso é um resultado da aplicação da modelagem ao recorte analisado, não um erro de execução.

Uma interpretação possível é que a fragmentação está associada a dois fatores:

- participação irregular, com muitos participantes tendo poucos palpites válidos;
- penalização por cobertura, que reduz a similaridade final quando dois participantes compartilham poucos jogos comparáveis.

O experimento com cobertura suavizada mostrou que reduzir a rigidez dessa penalização aumenta bastante a conectividade da rede. Portanto, a topologia é sensível à forma como a cobertura entra na fórmula.

PageRank deve ser interpretado como centralidade estrutural: participantes com alto PageRank estão em posições mais centrais na rede de similaridade. Isso não significa maior pontuação tradicional no bolão.

Comunidades indicam subgrupos mais densamente conectados por similaridade de palpites. Elas não indicam amizade, torcida, cópia ou causalidade.

## 15. Privacidade

O repositório foi organizado para não incluir dados pessoais identificáveis.

Participantes são anonimizados como:

```txt
P001, P002, P003, ...
```

Dados pessoais como nomes, e-mails, `avatarUrl`, `googleId`, tokens, hashes ou IDs reais não devem aparecer em CSVs públicos, gráficos, relatórios ou documentação.

O uso público do projeto deve partir de:

- `data/example/`, base fictícia segura;
- `data/processed/`, base real anonimizada segura;
- CSVs canônicos anonimizados próprios.

`data/raw/`, `data/private/` e arquivos `*.sql` continuam privados e não devem ser versionados.

## 16. Limitações

- Os resultados dependem do limiar `theta`.
- Os resultados dependem da função de cobertura.
- Muitos participantes podem ter poucos palpites válidos.
- Dados brutos privados não são distribuídos publicamente.
- Comunidades não indicam amizade, torcida, cópia ou causalidade.
- PageRank não mede desempenho tradicional.
- Grafos de visualização usam filtros para legibilidade.
- O projeto é um pipeline analítico offline, não um sistema de produção.

## 17. Testes

Execute:

```bash
pytest
```

Estado atual da suíte:

```txt
95 testes passando
```
