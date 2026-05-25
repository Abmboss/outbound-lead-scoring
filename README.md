# outbound-lead-scoring

> Pipeline de propensão para priorização de leads em operações outbound — XGBoost · BigQuery · Dataform · Vertex AI.

---

## Visão Geral

Equipes de outbound em seguros e serviços financeiros enfrentam um problema fundamental de priorização: uma lista de milhares de leads, mas capacidade para contatar apenas uma fração por dia. Ordenação aleatória ou por FILA desperdiça o tempo dos corretores em contatos de baixa probabilidade e deixa leads de alta intenção esfriarem.

Este projeto implementa um **sistema de propensão ponta a ponta** que ordena leads outbound pela probabilidade estimada de conversão. O output é consumido diretamente pelas filas do CRM, entregando aos corretores uma lista de chamadas priorizada e segmentada toda manhã.

**Saídas por lead:**
- `propensity_score` — probabilidade calibrada de conversão (0–1)
- `lead_tier` — segmento HOT / WARM / COLD com base nos limiares T50/T90/T95
- `top_arguments` — códigos de argumentação personalizados
- `broker_priority_rank` — posição do lead dentro do portfólio de cada corretor
- `suggested_call_window` — janela sugerida de ligação (MANHÃ / TARDE / QUALQUER) baseada em padrões históricos de atendimento

---

## Arquitetura

```
Fontes Brutas              Feature Store              Camada de Modelo        Exportação CRM
─────────────              ─────────────              ────────────────        ──────────────
Eventos CRM   ──▶  SQL BigQuery    ──▶  feat_behavioral   ──▶  XGBoost +     ──▶  mart_lead_scoring
Produtos      ──▶  (behavioral_      feat_rfm               Calibração            (Dataform SQLX)
Atributos CRM      features.sql)      feat_products         (Vertex AI)
                                      feat_crm             │
                                                           ▼
                                                    Limiares T50/T90/T95
                                                    → segmentação de tier
```

**Fluxo de dados:**
1. Eventos CRM e dados de produtos chegam no BigQuery (particionados por data)
2. Pipeline diário do Dataform computa as tabelas de features (`feat_*`) via agregações em janelas deslizantes
3. Job de predição em lote no Vertex AI pontua todos os leads ativos com o pipeline XGBoost treinado
4. `stg_propensity_scores` valida e segmenta os scores brutos
5. `mart_lead_scoring` junta scores com argumentação, deduplicação e contexto de CRM
6. Dashboard Power BI e sincronização via API do CRM consomem o mart

---

## Métricas T50 / T90 / T95

Limiares de classificação padrão (corte em 0,5) são inadequados para listas outbound com taxa de conversão base de 3–5%. Em vez disso, este projeto utiliza **métricas de percentil T**:

| Métrica | Definição | Significado operacional |
|---------|-----------|------------------------|
| **T50** | Score mínimo que captura 50% dos conversores | Contatar os X% melhores leads para alcançar metade de todos que converteriam |
| **T90** | Score mínimo que captura 90% dos conversores | Orçamento de contato "seguro" — captura quase todos os conversores com a menor lista |
| **T95** | Score mínimo que captura 95% dos conversores | Captura exaustiva — usado em campanhas de produtos de alto valor |

**Exemplo de interpretação:** Se T90 = 0,42 e cobre 28% da base de leads, contatar apenas os 28% melhores captura 90% das conversões. Isso se traduz diretamente em **ganhos de eficiência de contato** reportáveis para o negócio.

Os limiares são recomputados a cada retreinamento do modelo e armazenados em `stg_propensity_scores.sqlx` para segmentação de tier.

---

## Estrutura do Projeto

```
outbound-lead-scoring/
│
├── models/
│   └── propensity_model.py        # Pipeline XGBoost, calibração, cálculo de métricas T
│
├── features/
│   ├── feature_engineering.py     # Construtores Python de features (RFM, comportamental, produto, CRM)
│   └── sql/
│
├── sql/features/
│   └── behavioral_features.sql    # Agregações comportamentais em janelas deslizantes (7d/30d/90d)
│
├── dataform/definitions/
│   ├── staging/
│   │   └── stg_propensity_scores.sqlx   # Valida output do Vertex AI e atribui tiers
│   └── marts/
│       └── mart_lead_scoring.sqlx       # Mart final pronto para CRM com dedup + argumentação
│
├── data/
│   └── generate_sample.py         # Gera dataset sintético para desenvolvimento local
│
└── requirements.txt
```

---

## Início Rápido

### 1. Instalar dependências

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Gerar dados sintéticos

```bash
python data/generate_sample.py
# Cria: data/sample/leads.csv, events.csv, products.csv, crm.csv
```

### 3. Construir features e treinar

```python
import pandas as pd
from features.feature_engineering import build_feature_matrix
from models.propensity_model import PropensityModel, PropensityConfig

# Carrega dados de exemplo
df_leads    = pd.read_csv("data/sample/leads.csv")
df_events   = pd.read_csv("data/sample/events.csv")
df_products = pd.read_csv("data/sample/products.csv")
df_crm      = pd.read_csv("data/sample/crm.csv")

# Constrói matriz de features
df_features = build_feature_matrix(df_leads, df_events, df_products, df_crm)

# Treina o modelo
config = PropensityConfig(scale_pos_weight=24)  # razão negativos/positivos
model = PropensityModel(config=config)
model.fit(df_features, label_col="converted")

# Inspeciona limiares T
print(model.thresholds.summary())

# Pontua e segmenta novos leads
scored = model.score_and_tier(df_features)
print(scored.head())

# Salva o modelo
model.save("artifacts/model_v1")
```

### 4. Avaliar no holdout

```python
metrics = model.evaluate(df_holdout, label_col="converted")
# {'auroc': 0.82, 'aucpr': 0.31, 'base_rate': 0.04, 'lift_top10': 4.7}
```

---

## Decisões de Design

### Calibração de probabilidade
Scores brutos do XGBoost são bem ordenados mas mal calibrados — um output de 0,7 não significa "70% de chance de converter". A regressão isotônica via `CalibratedClassifierCV` corrige isso, tornando os limiares T operacionalmente significativos e estáveis entre retreinamentos.

### Deduplicação
O mesmo cliente pode aparecer em múltiplos pipelines do CRM (retenção, cross-sell, outbound frio). O `mart_lead_scoring.sqlx` deduplica por CPF/CNPJ, mantendo apenas a ocorrência com maior score. Sem isso, corretores de filas diferentes contatariam o mesmo cliente, degradando a experiência e inflando contagens.

### `scale_pos_weight`
Com taxa base de 3–5%, o XGBoost ingênuo ignora a classe minoritária. `scale_pos_weight ≈ razão neg/pos` (~20–30x) força o modelo a penalizar mais os falsos negativos — o tradeoff correto para outbound, onde conversores perdidos são caros.

### Janelas deslizantes (7d / 30d / 90d)
Janelas curtas (7d) capturam sinais de recência — um lead que clicou em um e-mail ontem está quente. Janelas longas (90d) capturam frequência e engajamento histórico. Usar as três simultaneamente permite que o modelo aprenda padrões temporais distintos sem viés de seleção de features.

### AUCPR em vez de AUROC
Com forte desbalanceamento de classes, o AUROC é otimista (dominado pelos verdadeiros negativos). A Área sob a Curva Precision-Recall (AUCPR) é a métrica correta — mede diretamente o desempenho na classe minoritária que importa.

---

## Integração BigQuery / Dataform

O modelo Python roda offline (local ou Vertex AI). A camada BigQuery/Dataform é responsável por:

- **Computação de features em escala** — `behavioral_features.sql` processa milhões de eventos diariamente com queries com pruning de partição, evitando full table scans
- **Validação de scores** — `stg_propensity_scores.sqlx` rejeita linhas corrompidas e garante limites de score antes de propagar downstream
- **Enriquecimento CRM** — `mart_lead_scoring.sqlx` é a fonte única da verdade consumida pelo Power BI e sincronização via API do CRM

Para implantar o pipeline Dataform:
```bash
dataform init bigquery --project-id SEU_PROJETO --location us-east1
dataform run --tags lead_scoring
```

---

## Contribuindo

1. Faça um fork do repositório
2. Crie uma branch: `git checkout -b feat/sua-feature`
3. Faça as alterações e abra um Pull Request contra `main`

Estilo de código: `black` + `ruff`. SQL: palavras-chave em maiúsculo, indentação de 4 espaços, uma CTE por etapa lógica.

---

## Licença

MIT License. Veja [LICENSE](LICENSE) para detalhes.
