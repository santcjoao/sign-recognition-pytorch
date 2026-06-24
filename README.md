# GTSRB — Classificação de Placas de Trânsito com CNNs

Pipeline modular em PyTorch para classificação de imagens do dataset **GTSRB** (German Traffic Sign Recognition Benchmark — 43 classes).

---

## Arquitetura

```text
experiments.py          # Ponto de entrada — declara configs e executa experimentos
plots.py                # Gera gráficos e exporta CSV final após os runs
│
├── src/
│   ├── core/types.py   # Dataclasses imutáveis (ModelConfig, Hyperparameters, ...)
│   ├── models/
│   │   ├── baseline.py # CNN simples: Conv→ReLU→Pool ×2 → FC ×2
│   │   ├── custom.py   # CNN profunda: 3-4 blocos conv, Dropout, canais configuráveis
│   │   └── factory.py  # build_model() — resolve config.name → nn.Module
│   ├── pipe/
│   │   ├── pipeline.py # Orquestrador: carrega dados → treina → testa → salva resultados
│   │   ├── training.py # Loop de treino/validação com early stopping por val_acc
│   │   └── testing.py  # Inferência no conjunto de teste
│   └── utils/
│       ├── gtsrb.py    # Arquivo do professor (NÃO modificar)
│       └── args.py     # Parser de argumentos CLI
│
├── data/               # Dataset GTSRB (download automático)
└── results/            # Saídas: pesos (.pth), histórico (.json), predições (.csv)
```

O `Pipeline` recebe uma suíte de experimentos (lista de configs imutáveis) e, para cada um, executa: **carregamento de dados → treino → teste → serialização em disco**. `Trainer` e `Tester` operam apenas em memória; toda persistência fica no `Pipeline`.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# Treinar modelos individuais
python experiments.py --baseline
python experiments.py --custom

# Experimentos comparativos
python experiments.py --compare-optimizers   # SGD / SGDm / Adam (baseline + BN)
python experiments.py --compare-norm         # Sem vs Com Batch Normalization

# Combinar flags livremente
python experiments.py --baseline --compare-optimizers --compare-norm

# Gerar gráficos e CSV final (após rodar experimentos)
python plots.py
```

O dataset é baixado automaticamente na primeira execução.

---

## Saídas do `plots.py`

**Por experimento** (em `results/runs/<exp_name>/`):

| Arquivo | Descrição |
|---|---|
| `curves_plot.png` | Curvas de loss e acurácia ao longo das épocas |
| `confusion_matrix.png` | Matriz de confusão completa (43×43) |
| `easiest_hardest_classes.png` | Top 5 classes mais fáceis e mais difíceis |
| `acc_vs_support.png` | Scatter: acurácia × nº de exemplos no teste |
| `all_class_accuracies.png` | Barplot com acurácia de todas as 43 classes |
| `prediction_examples.png` | Grid de 5 acertos e 20 erros com imagens reais |

**Comparativos** (em `results/comparatives/`):

| Arquivo | Descrição |
|---|---|
| `compare_optimizers.png` | Acurácia média por otimizador |
| `compare_batchnorm.png` | Impacto do Batch Normalization |
| `compare_augmentation.png` | Impacto do Data Augmentation |

**Outros:**
- `results/metrics_summary.csv` — tabela consolidada ordenada por desempenho.
- `JOAO_NATHAN_THIAGO_RESULTS.csv` — submissão final com predições do melhor modelo, no formato de `save_predictions`.
