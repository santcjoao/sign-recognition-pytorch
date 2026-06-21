# GTSRB Machine Learning Pipeline

Este repositório contém um pipeline modular desenvolvido em PyTorch para a classificação do conjunto de dados GTSRB (German Traffic Sign Recognition Benchmark).

O projeto adota o padrão de Especificação Executável, em que as configurações de experimentos e transformações são declaradas de forma imutável, enquanto o pipeline orquestra a execução dos estágios de treino e teste e centraliza as operações de leitura e escrita.

---

## Arquitetura do Projeto

```text
├── data/                  # Dados do GTSRB (download automático)
├── results/               # Saída de resultados (logs, pesos e predições)
├── src/
│   ├── core/types.py      # Dataclasses de configuração e resultados
│   ├── models/baseline.py # Arquitetura Baseline CNN parametrizável
│   ├── pipe/              # Módulos de orquestração, treino e teste
│   └── utils/gtsrb.py     # Utilitários obrigatórios fornecidos (seed=42)
├── experiments.py         # Declaração e execução de suítes de testes
├── pyproject.toml         # Configuração centralizada para ferramentas Ruff e Mypy
└── requirements.txt       # Lista de dependências de software
```

---

## Requisitos Acadêmicos Atendidos

* **Separação de Dados:** Divisão determinística (80/20) realizada via `get_dataloaders()` de `src/utils/gtsrb.py` (semente de validação fixa em 42).
* **Desacoplamento de I/O:** Os módulos `Trainer` e `Tester` realizam operações exclusivamente em memória. A persistência em disco (pesos `.pth`, logs em `.json` e predições em `.csv`) é realizada unicamente pelo módulo `Pipeline`.
* **Exportação de Predições:** Geração automatizada do arquivo CSV de predições conforme formato especificado pela função utilitária `save_predictions`.
* **Modelo Parametrizável:** Rede sequencial flexível (`Conv-ReLU-Pool-Conv-ReLU-Pool-Linear-ReLU-Linear`) com suporte a Batch Normalization e Dropout configuráveis e cálculo automático de dimensões.

---

## Configuração e Execução

### 1. Configurar o Ambiente
Para inicializar o ambiente virtual e instalar as dependências necessárias de acordo com o sistema operacional:

#### Windows:
Execute os seguintes comandos no terminal (PowerShell ou Prompt de Comando) na raiz do projeto:
```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux / macOS:
Execute os seguintes comandos no terminal na raiz do projeto:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar Experimentos
Configure a suíte e a transformação a serem avaliadas nas constantes ao final do arquivo `experiments.py` e execute o script de entrada:
```bash
python3 experiments.py
```
```python
# Configuração em experiments.py
CHOOSEN_SUITE = BASELINE_SUITE
CHOOSEN_TRANSFORM = IDENTITY_TRANSFORM
```
