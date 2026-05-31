# Churn-Radar

Modelagem preditiva de churn em telecom. Comparação de 3 modelos de ML com análise de
interpretabilidade e métricas de negócio para estratégia de retenção de clientes.

## Dataset

[Telecom Churn](https://www.kaggle.com/datasets/barun2104/telecom-churn) (`barun2104/telecom-churn`):
~3.333 clientes, 10 features numéricas e o alvo binário `Churn` (~14,5% positivos).
O acesso é feito via `kagglehub`.

## Modelos selecionados

Escolhidos com a skill **/model-selector** (classificação binária, dados pequenos e
desbalanceados, interpretabilidade relevante, prioridade balanceada):

| Modelo | Papel |
|---|---|
| **Logistic Regression** | Baseline interpretável (odds ratios), `class_weight='balanced'` |
| **Random Forest** | Robusto, não-linear, `class_weight='balanced'` |
| **XGBoost** | Teto de performance em dados tabulares, `scale_pos_weight` |

## Estrutura

```
src/
  config.py     # caminhos, semente, parâmetros de negócio (CLV, custos)
  data.py       # download via kagglehub + carga do CSV
  features.py   # split treino/teste estratificado
  train.py      # treina e salva os 3 modelos
  evaluate.py   # ROC-AUC, PR-AUC, precision/recall/F1 e gap treino-teste (overfit)
  tune.py       # tuning de hiperparâmetros por CV (RandomizedSearchCV)
  interpret.py  # SHAP (RF/XGB) e coeficientes (LogReg)
  business.py   # threshold ótimo e ROI de retenção
notebooks/
  churn_analysis.ipynb   # EDA + SHAP + métricas de negócio
```

## Setup

```bash
pip install -r requirements.txt          # kagglehub fica pinado em 1.0.0
cp .env.example .env                      # preencha KAGGLE_USERNAME e KAGGLE_KEY
```

O arquivo `.env` é ignorado pelo git (ver `.gitignore`) e **nunca** deve ser commitado.

## Como rodar

```bash
python -m src.train          # treina os 3 modelos (baseline) e imprime a comparação
python -m src.train smote    # treina com oversampling SMOTE (salva *_smote.joblib)
python -m src.train compare  # baseline vs SMOTE lado a lado (foco no recall)
python -m src.tune           # tuning por CV (RandomizedSearchCV) -> *_tuned.joblib
python -m src.interpret  # SHAP/coeficientes dos modelos salvos (gera models/shap_*.png)
python -m src.business   # ROI de retenção por modelo (gera models/profit_*.png)
jupyter notebook notebooks/churn_analysis.ipynb   # EDA, SHAP e ROI
```

> **Nota:** o acesso ao Kaggle costuma estar bloqueado em ambientes de nuvem. O download
> dos dados e o treino completo devem ser executados **localmente** com um `.env` válido.

## Resultados

Hiperparâmetros otimizados por validação cruzada (`RandomizedSearchCV`, PR-AUC).
Desempenho no conjunto de teste:

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 | Gap treino-teste |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.857 | **0.730** | 0.540 | **0.773** | 0.636 | 0.116 |
| **Random Forest** | 0.856 | 0.705 | 0.497 | 0.753 | 0.598 | 0.095 |
| Logistic Regression | 0.813 | 0.419 | 0.313 | 0.804 | 0.451 | 0.011 |

- **XGBoost e Random Forest** lideram em PR-AUC (métrica-chave em dados desbalanceados),
  com recall ~0,76 — pegam a maioria dos churners.
- **Logistic Regression** generaliza muito bem (gap ~0), mas com PR-AUC bem menor:
  ótima como baseline interpretável.

### Métricas de negócio — ROI da campanha de retenção

O modelo dá a cada cliente uma **probabilidade de churn** (de 0 a 1). O *threshold* é o
**ponto de corte** dessa probabilidade a partir do qual decidimos agir: clientes com
probabilidade **acima** do threshold entram na campanha de retenção; os demais, não.
Threshold baixo = aborda quase todo mundo (gasta muito à toa); threshold alto = aborda só
os casos mais prováveis (pode deixar churners passarem). O **threshold ótimo** é o corte que
gera o maior lucro.

Otimizando esse corte para maximizar o lucro líquido esperado
(`CLV=1.000`, custo da ação `=100`, sucesso `=30%` — parâmetros em `src/config.py`).
Agir sobre **todos** os clientes (baseline, sem modelo) daria **prejuízo de R$ 37.600**;
o modelo transforma esse prejuízo em lucro:

| Modelo | Threshold ótimo | Clientes-alvo | Lucro no ótimo | Uplift vs. baseline |
|---|---|---|---|---|
| **XGBoost** | 0,71 | 99 | **R$ 10.500** | **+R$ 48.100** |
| Random Forest | 0,62 | 100 | R$ 10.100 | +R$ 47.700 |
| Logistic Regression | 0,73 | 45 | R$ 2.400 | +R$ 40.000 |

**XGBoost e Random Forest rendem ~4× mais lucro que a Logistic Regression** e têm curvas
com platô largo (≈0,5–0,9), ou seja, o lucro se mantém mesmo com variações no threshold
operacional — robustez importante para produção. Curvas completas em `models/profit_*.png`.
