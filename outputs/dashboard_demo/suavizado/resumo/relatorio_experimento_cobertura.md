# Relatório do Experimento Complementar  Cobertura Suavizada

## Parâmetros
- `theta`: 0.2
- `gamma`: 0.5
- Fórmula experimental: `sim_final = sim_media * cobertura^gamma`.
- Modelo oficial preservado: cobertura linear com `gamma = 1.0`.

## Principais números
- Vértices: 136
- Arestas: 1961
- Componentes conexas: 27
- Maior componente: 110
- Participante mais central: `P015`

## Comparação com o modelo linear
- `cobertura_linear`: 515 arestas, 93 componentes, 92 isolados, maior componente com 44 participantes.
- `cobertura_suavizada`: 1961 arestas, 27 componentes, 26 isolados, maior componente com 110 participantes.

## Comunidades da maior componente
- Comunidade 1: 52 participantes, participante central `P015`.
- Comunidade 2: 48 participantes, participante central `P111`.
- Comunidade 3: 8 participantes, participante central `P013`.
- Comunidade 4: 2 participantes, participante central `P132`.

## Principais vértices-ponte
- `P015`: comunidade 1, grau intercomunidades 43, força intercomunidades 15.6373.
- `P099`: comunidade 1, grau intercomunidades 40, força intercomunidades 13.9012.
- `P092`: comunidade 1, grau intercomunidades 43, força intercomunidades 14.2202.
- `P126`: comunidade 1, grau intercomunidades 39, força intercomunidades 14.3334.
- `P007`: comunidade 1, grau intercomunidades 42, força intercomunidades 13.8952.
- `P013`: comunidade 3, grau intercomunidades 69, força intercomunidades 25.7220.
- `P_TESTE`: comunidade 1, grau intercomunidades 39, força intercomunidades 14.2321.
- `P081`: comunidade 1, grau intercomunidades 36, força intercomunidades 12.2091.

## Interpretação crítica
- A rede suavizada é muito menos fragmentada que a rede com cobertura linear.
- Isso indica que a topologia é sensível à penalização da cobertura.
- O resultado deve ser apresentado como análise complementar, não como substituição automática do modelo principal.
- Nas visualizações, `weight` representa similaridade e controla espessura de arestas; `distance = 1 - weight` é usado em caminhos mínimos e betweenness.
