# Relatório Final  Rede de Similaridade de Palpites

## 1. Dados do experimento
- `participantes_validos`: 135
- `jogos_validos`: 33
- `palpites_validos`: 1113
- `resultados_validos`: 33
- `media_palpites_por_participante`: 8.24444
- `mediana_palpites_por_participante`: 4
- `min_palpites_por_participante`: 1
- `max_palpites_por_participante`: 33
- `participantes_com_1_palpite`: 25
- `participantes_com_ate_4_palpites`: 78

## 2. Parâmetros da análise
- Limiar experimental da rede final: `theta = 0.2`.
- Referência estrita preservada no projeto: `THETA_SIMILARIDADE = 0.50`.
- A similaridade entre participantes usa média por jogos comparáveis ajustada por cobertura.

## 3. Métricas globais do grafo
- `vertices`: 135
- `arestas`: 479
- `densidade`: 0.0529574
- `componentes`: 93
- `maior_componente`: 43
- `grau_medio`: 7.0963
- `forca_media`: 2.24444
- `peso_medio_arestas`: 0.316284
- `agrupamento_medio`: 0.121713
- `distancia_media_maior_componente`: 1.04195
- `diametro_maior_componente`: 2.37121

## 4. Componentes conexas
- Tamanho 1: 92 componente(s), 92 participante(s).
- Tamanho 43: 1 componente(s), 43 participante(s).

## 5. Comunidades da maior componente
- Comunidade 1: 24 participantes, 196 arestas internas, participante mais central `P126` (PageRank 0.0320595).
- Comunidade 2: 19 participantes, 78 arestas internas, participante mais central `P013` (PageRank 0.0312834).

## 6. Ranking de centralidade
- 1. `P126` PageRank 0.0320595, grau 36, força 13.0814.
- 2. `P013` PageRank 0.0312834, grau 36, força 13.2121.
- 3. `P015` PageRank 0.0311922, grau 32, força 13.7216.
- 4. `P099` PageRank 0.0306096, grau 33, força 12.0284.
- 5. `P116` PageRank 0.0289703, grau 33, força 12.5511.
- O PageRank representa centralidade estrutural na rede de similaridade, não pontuação tradicional do bolão.

## 7. Figuras finais
- `figuras/analise_sensibilidade_limiar.png`: gerada.
- `figuras/distribuicao_palpites_participante.png`: gerada.
- `figuras/distribuicao_tamanho_componentes_melhorado.png`: gerada.
- `figuras/tamanho_comunidades_maior_componente.png`: gerada.
- `figuras/grafo_maior_componente_comunidades.png`: gerada.
- `figuras/grafo_maior_componente_filtrado.png`: gerada.
- `figuras/heatmap_similaridade_ordenado.png`: gerada.
- `figuras/ranking_centralidade_pagerank_melhorado.png`: gerada.

## 8. Interpretação crítica
- A rede final usa `theta = 0.20`, escolhido por análise de sensibilidade.
- O modelo preserva `THETA_SIMILARIDADE = 0.50` como referência estrita.
- A rede real ficou fragmentada.
- Há 92 componentes unitárias e uma maior componente com 43 participantes.
- A maior componente contém duas comunidades principais.
- A fragmentação está associada à participação irregular e à penalização por cobertura.
- PageRank mede centralidade estrutural, não pontuação tradicional do bolão.

## 9. Limitações
- A análise usa apenas jogos encerrados com resultado e pelo menos um palpite.
- Participantes com poucos palpites tendem a ter baixa cobertura nas comparações.
- Comunidades representam estrutura de similaridade, não relações sociais ou causalidade.
- O grafo filtrado para visualização não altera as métricas calculadas no grafo final.

## 10. Arquivos gerados
- `graphml`: `outputs\final_theta_0_20\grafos\grafo_final.graphml`
- `json_grafo`: `outputs\final_theta_0_20\grafos\grafo_final.json`
- `relatorio_md`: `outputs\final_theta_0_20\resumo\relatorio_final.md`
- `relatorio_json`: `outputs\final_theta_0_20\resumo\relatorio_final.json`
- `validacao_txt`: `outputs\final_theta_0_20\resumo\validacao_final.txt`
- `validacao_json`: `outputs\final_theta_0_20\resumo\validacao_final.json`
