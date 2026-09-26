# Detecção de Fraude em Transações de Cartão de Crédito

Projeto do desafio de Machine Learning da DIO: treinar modelos que detectam
fraude em transações reais de cartão de crédito, um problema em que a fraude
é rara e a acurácia engana.

O notebook completo está neste repositório, com as saídas salvas (tabelas e
gráficos) como evidência de que roda do início ao fim.

## 1. O problema e por que o desbalanceamento muda a avaliação

A base contém transações reais de cartão de crédito, com as colunas `Time`,
`Amount`, `Class` e as variáveis `V1` a `V28` (transformadas por PCA para
proteger a privacidade dos clientes).

O ponto central é o desbalanceamento: cerca de 99,8% das transações são
normais e apenas 0,17% são fraude. Um modelo que responde "não é fraude"
para tudo acerta quase sempre — acurácia de 99,8% — e não detecta nenhuma
fraude. Por isso a avaliação sai da acurácia e vai para as métricas da
classe de fraude:

- **Recall** (métrica principal): de todas as fraudes reais, quantas o
  modelo pegou. Perder fraude é o erro mais caro.
- **Precisão**: de tudo que o modelo marcou como fraude, quanto era fraude
  de verdade. Marcar cliente honesto como fraude também tem custo.
- **F1-Score**: equilíbrio entre as duas.

## 2. Preparação dos dados

- Carregamento do dataset direto pelo link (o arquivo não fica no repositório);
- Análise da proporção de fraudes com `value_counts(normalize=True)`;
- Feature engineering: criação de `Amount_log` com `np.log1p` para reduzir
  o efeito de valores extremos no valor da transação;
- Padronização de `Amount` com `StandardScaler`;
- Separação treino/teste com `train_test_split(stratify=y, test_size=0.3,
  random_state=42)`, para manter a proporção de fraudes igual nos dois
  conjuntos;
- Tratamento do desbalanceamento por três vias:
  - **Undersampling**: redução aleatória das transações normais para
    igualar o número de fraudes;
  - **Oversampling com SMOTE**: geração de exemplos sintéticos da classe
    de fraude;
  - **Peso de classes** dentro dos próprios modelos
    (`class_weight='balanced'` no Random Forest e `scale_pos_weight=10`
    no XGBoost).

## 3. Comparação entre os modelos

Comecei pela regressão logística como baseline e comparei com modelos mais
fortes, sempre olhando para a classe de fraude (1):

| Modelo | Recall (fraude) | Precisão (fraude) | F1 (fraude) |
|---|---|---|---|
| Regressão Logística (baseline) | 0.66 | 0.84 | 0.73 |
| Random Forest (class_weight='balanced') | 0.76 | 0.84 | 0.79 |
| XGBoost (scale_pos_weight=10) | 0.78 | 0.94 | 0.85 |

Cada modelo avançou no recall da fraude: 0.66 no baseline → 0.76 no
Random Forest → 0.78 no XGBoost, que também elevou a precisão para 0.94 —
o melhor resultado geral.

Os hiperparâmetros do XGBoost foram ajustados com `GridSearchCV` usando
`scoring='recall'` (melhor combinação: `max_depth=5`, `n_estimators=100`).

A leitura do relatório de classificação mostra exatamente o problema do
desbalanceamento: a acurácia geral fica em 1.00, mas o recall da fraude no
baseline foi de apenas 0.66 — ou seja, 1 em cada 3 fraudes passou.

## 4. Limiar de decisão e SHAP

- **Limiar de decisão**: em vez do padrão 0.5, o limiar foi ajustado para
  **0.3**. Com o limiar mais baixo, o modelo marca como fraude transações
  com probabilidade a partir de 30%, o que aumenta o recall — captura mais
  fraudes reais — ao custo de uma precisão menor, que é o trade-off aceito
  num problema em que perder fraude é o erro mais caro. O relatório de
  classificação com o limiar de 0.3 está salvo no notebook.
- **SHAP**: usei `shap.Explainer` no XGBoost e o gráfico de barras mostra
  quais variáveis mais pesaram nas decisões. As três variáveis de maior
  impacto foram **V14** (valor SHAP médio +1.43), **V12** (+0.52) e
  **V10** (+0.40) — todas derivadas do PCA, o que indica que o modelo
  aprendeu padrões reais das transações fraudulentas, e não ruído.

## 5. O que mudei em relação ao que a Expert fez

- Apliquei **três** formas de tratar o desbalanceamento (undersampling
  manual, SMOTE e peso de classes nos modelos) para comparar as
  alternativas, como o desafio sugere;
- Usei **peso de classes** no Random Forest (`class_weight='balanced'`) e
  no XGBoost (`scale_pos_weight=10`), em vez de depender só da
  reamostragem;
- Adicionei **ajuste de hiperparâmetros com GridSearchCV** otimizando
  diretamente o recall;
- Organizei a comparação dos modelos em tabela, com recall, precisão e F1
  da classe de fraude lado a lado.

## Como rodar

1. Abra o notebook `Cópia_de_DesafioPython_DIO02.ipynb` no Google Colab;
2. Execute todas as células na ordem — o dataset é baixado direto pelo link;
3. Bibliotecas necessárias: `pandas`, `numpy`, `scikit-learn`, `imblearn`,
   `xgboost`, `shap`, `matplotlib`.
