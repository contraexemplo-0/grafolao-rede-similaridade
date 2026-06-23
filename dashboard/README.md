# Dashboard local  Laços de Palpite

Este dashboard é um protótipo local em Streamlit para demonstrar uma experiência de produto do módulo de rede de similaridade do Grafolão.

Ele não possui:

- login real;
- banco de dados;
- integração com o sistema Grafolão;
- dados pessoais;
- conexão externa.

O usuário fixo da demonstração é `P_TESTE`, um participante artificial criado apenas para fins de navegação no dashboard.

## Estrutura

```txt
dashboard/
  app.py
  gerar_dados_demo.py
  gerar_resultados_demo.py
  requirements.txt
  data/
    demo/
  assets/
  styles/
```

## 1. Instalar dependências do dashboard

O projeto principal não depende de Streamlit. Para rodar o protótipo visual:

```bash
pip install -r dashboard/requirements.txt
```

## 2. Gerar dados demo com P_TESTE

O script abaixo lê os CSVs anonimizados em `data/processed/` e cria uma cópia segura em `dashboard/data/demo/`, adicionando o participante artificial `P_TESTE`.

```bash
python dashboard/gerar_dados_demo.py
```

O script:

- não altera `data/processed/`;
- não acessa `data/raw/` ou `data/private/`;
- não expõe dados pessoais;
- mantém o formato dos CSVs canônicos;
- cria palpites plausíveis para `P_TESTE` em quase todos os jogos válidos.

Arquivos gerados:

```txt
dashboard/data/demo/participantes.csv
dashboard/data/demo/jogos.csv
dashboard/data/demo/palpites.csv
dashboard/data/demo/resultados.csv
```

## 3. Gerar resultados do dashboard

Depois de gerar os dados demo:

```bash
python dashboard/gerar_resultados_demo.py
```

Esse comando gera:

```txt
outputs/dashboard_demo/linear/
outputs/dashboard_demo/suavizado/
```

Configurações:

- modelo linear: `theta = 0.20`, `gamma = 1.0`;
- modelo suavizado: `theta = 0.20`, `gamma = 0.5`.

Esses outputs são separados dos resultados consolidados do artigo.

## 4. Rodar o dashboard

```bash
streamlit run dashboard/app.py
```

O dashboard abre com:

- usuário atual fixo: `P_TESTE`;
- seletor de modelo: cobertura linear ou cobertura suavizada;
- páginas de visão geral, semelhantes, grupos, ranking e mapa da rede.

## Mapa da rede interativo

A página **Mapa da rede** usa Plotly + NetworkX para exibir uma rede explorável com zoom, pan e hover.

Modos disponíveis:

- **Minha rede**: mostra `P_TESTE` e seus participantes mais semelhantes.
- **Maior componente**: mostra a maior componente conexa do modelo selecionado.
- **Rede global simplificada**: mostra uma versão filtrada para manter legibilidade.

Codificação visual:

- nó = participante;
- aresta = relação de similaridade;
- cor = comunidade;
- tamanho = centralidade;
- destaque amarelo = `P_TESTE`.

Se o Plotly ainda não estiver instalado, a página mostra uma mensagem amigável e usa imagem estática como fallback quando disponível.

## Arquivos esperados

O dashboard lê principalmente:

```txt
outputs/dashboard_demo/linear/tabelas/
outputs/dashboard_demo/linear/figuras/
outputs/dashboard_demo/suavizado/tabelas/
outputs/dashboard_demo/suavizado/figuras/
```

Se algum arquivo não existir, a interface mostra aviso amigável e instruções para gerar os dados.

## Observações de privacidade

- `P_TESTE` é artificial.
- Os dados usados são anonimizados.
- O dashboard não usa dump SQL privado.
- O dashboard não acessa banco de dados.
- O dashboard é apenas uma demonstração local de produto.
