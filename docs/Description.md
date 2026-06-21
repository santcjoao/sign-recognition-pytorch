# TRABALHO DE INTELIGENCIA ARTIFICIAL - GTSRB

## 1. OBJETIVO
Neste trabalho, vocês deverão desenvolver e avaliar modelos de redes neurais convolucionais para classificação de imagens do dataset GTSRB - German Traffic Sign Recognition Benchmark. O objetivo principal é obter alta acurácia global, mas também investigar o comportamento do modelo por classe. O grupo deverá identificar quais classes são mais difíceis, analisar possíveis causas dos erros e propor experimentos para melhorar o desempenho dessas classes.

Ao final do trabalho, o grupo deverá ser capaz de responder perguntas como:
* Quais classes foram mais fáceis e mais difíceis?
* O modelo erra mais em classes com poucos exemplos?
* Quais classes são confundidas entre si?
* Batch normalization melhora o treinamento?
* SGD, SGD com momentum e Adam produzem resultados diferentes?
* Data augmentation melhora a acurácia global ou apenas algumas classes?

## 2. DATASET
O dataset utilizado será o GTSRB, disponível no `torchvision.datasets.GTSRB`. O GTSRB contém imagens de placas de trânsito alemãs organizadas em 43 classes. As imagens possuem variações de tamanho, iluminação, posição, nitidez e perspectiva. O grupo deverá redimensionar as imagens para um tamanho fixo. A escolha do tamanho da imagem pode influenciar nos resultados.

O carregamento do dataset deve seguir a separação oficial:
* `split="train"` para treinamento e validação;
* `split="test"` para avaliação final.

O conjunto de treinamento deverá ser dividido em treino e validação. O conjunto de teste deve ser usado apenas para avaliação final. Todos os experimentos devem usar a mesma divisão treino/validação, para que os resultados sejam comparáveis.

## 3. TAREFA
A tarefa consiste em treinar modelos capazes de classificar corretamente a placa de trânsito presente em cada imagem. Cada grupo deverá implementar:
* Um modelo baseline;
* Experimentos de melhoria;
* Uma análise detalhada dos resultados por classe;
* Um arquivo `.csv` contendo as predições do melhor modelo no conjunto de teste.

## 4. MODELO BASELINE OBRIGATÓRIO
O grupo deve implementar uma CNN simples, treinada do zero, contendo pelo menos:
* camadas convolucionais;
* função de ativação ReLU;
* pooling;
* uma ou mais camadas fully connected;
* função de perda `CrossEntropyLoss`;
* cálculo de acurácia no conjunto de validação e no conjunto de teste.

O baseline deve servir como referência para todos os outros experimentos.

Exemplo de arquitetura baseline aceitável:
```text
Conv2d -> ReLU -> MaxPool
Conv2d -> ReLU -> MaxPool
Flatten
Linear -> ReLU
Linear -> saida com 43 classes
```

O grupo pode propor uma arquitetura baseline diferente, desde que ela seja simples, treinada do zero e claramente descrita no relatório.

## 5. EXPERIMENTOS OBRIGATÓRIOS
Cada grupo deverá realizar, no mínimo, os seguintes experimentos.

### 5.1 Experimento 1 - Otimizadores
Comparar pelo menos três estratégias de otimização:
* SGD;
* SGD com momentum;
* Adam.

### 5.2 Experimento 2 - Batch Normalization
Comparar uma arquitetura:
* sem batch normalization;
* com batch normalization.

### 5.3 Experimento 3 - Data Augmentation
Aplicar pelo menos duas técnicas de data augmentation, como:
* pequenos deslocamentos;
* rotações leves;
* alterações de brilho;
* alterações de contraste;
* random crop;
* blur leve.

## 6. EXPERIMENTOS OPCIONAIS
Além dos experimentos obrigatórios, o grupo pode realizar um ou mais experimentos extras.

### 6.1 Arquiteturas alternativas
Comparar a CNN baseline com:
* uma VGG-like pequena;
* uma ResNet pequena;
* uma arquitetura criada pelo próprio grupo;
* regularização com dropout.

Modelos pré-treinados não são permitidos. Todas as arquiteturas devem ser treinadas do zero.

### 6.2 Tamanho da imagem
Comparar o desempenho usando imagens redimensionadas para diferentes tamanhos. O grupo pode discutir o impacto do tamanho da imagem na acurácia, no custo computacional e no tempo de treinamento.

## 7. MÉTRICAS OBRIGATÓRIAS
O relatório deve apresentar as seguintes métricas:

| Métrica | Descrição |
| :--- | :--- |
| Acurácia global | Percentual total de imagens classificadas corretamente. |
| Acurácia por classe | Acurácia individual para cada uma das 43 classes. |
| Macro accuracy | Média das acurácias por classe. |
| Matriz de confusão | Mostra quais classes são confundidas entre si. |
| Curvas de treino | Loss e acurácia ao longo das épocas. |

## 8. SALVAMENTO DOS RESULTADOS
Além das métricas apresentadas no relatório, cada grupo deverá salvar as predições do melhor modelo em um arquivo `.csv`. Esse arquivo será utilizado pelo professor para calcular métricas padronizadas e comparar os resultados entre os grupos.

Para garantir uma comparação justa, todos os grupos deverão utilizar o arquivo auxiliar fornecido pelo professor para carregar os dados e salvar as predições. Esse arquivo não deve ser modificado.

O arquivo auxiliar define:
* o carregamento do dataset GTSRB;
* a divisão fixa entre treino e validação, usando `seed = 42`;
* a proporção de 80% para treino e 20% para validação;
* o uso do conjunto oficial de teste do GTSRB;
* o DataLoader de teste com `shuffle=False`;
* a função padronizada `save_predictions` para exportar o arquivo final.

Cada grupo deverá entregar obrigatoriamente um arquivo chamado: `NOME1_NOME2_RESULTS.csv` onde `NOME1` e `NOME2` devem ser substituídos pelo primeiro nome dos integrantes do grupo. Por exemplo: `ANA_PEDRO_RESULTS.csv`
Esse arquivo deve conter as predições do melhor modelo escolhido pelo grupo no conjunto de teste.

### 8.1 Formato obrigatório do arquivo
O arquivo de resultados deve ser gerado usando a função `save_predictions` fornecida no arquivo auxiliar. O arquivo deve conter uma linha para cada imagem do conjunto de teste, seguindo exatamente a ordem do DataLoader de teste. Como o conjunto oficial de teste do GTSRB possui 12.630 imagens, o arquivo final deve conter 12.630 linhas de predições, além do cabeçalho.

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| image_index | inteiro | Índice da imagem no conjunto de teste. |
| predicted_class | inteiro | Classe prevista pelo modelo, de 0 a 42. |

Exemplo de arquivo:
```csv
image_index, predicted_class
0,16
1,1
2,38
3,33
4,11
5,38
```

### 8.2 Exemplo de uso
Após obter as predições do modelo no conjunto de teste, o grupo deverá salvar o arquivo usando a função `save_predictions`.

```python
from gtsrb_utils import get_dataloaders, save_predictions

train_loader, val_loader, test_loader = get_dataloaders()
# Configuração recomendada: img_size = 32, batch_size = 128

# y_pred deve conter as classes previstas pelo modelo no conjunto de teste
# y_pred deve ter shape (12630,)
save_predictions(y_pred, "ANA_PEDRO_RESULTS.csv", experiment_name="CNN BatchNorm Adam Augmentation")
```

O vetor `y_pred` deve conter apenas as classes previstas pelo modelo, representadas por números inteiros de 0 a 42.

## 9. RELATÓRIO
O relatório deve conter as seguintes seções.

### 9.1 Introdução
* Descrição do problema;
* Breve descrição do dataset GTSRB;
* Objetivo do trabalho.

### 9.2 Metodologia
* Pré-processamento;
* Divisão treino/validação/teste;
* Arquiteturas utilizadas;
* Otimizadores;
* Hiperparâmetros;
* Estratégias de augmentation;
* Métricas usadas.

### 9.3 Resultados
* Tabelas de resultados;
* Curvas de treino;
* Acurácia por classe;
* Matriz de confusão;
* Exemplos de imagens classificadas corretamente;
* Exemplos de imagens classificadas incorretamente.

### 9.4 Discussão
* Quais classes foram mais difíceis?
* O que melhorou a acurácia global?
* O que melhorou a macro accuracy?
* O que aconteceu com a pior classe?
* Houve trade-off entre melhorar classes difíceis e manter acurácia global?
* Quais erros do modelo parecem mais compreensíveis?
* Quais erros foram inesperados?

### 9.5 Conclusão
* Melhor modelo encontrado;
* Principais aprendizados;
* Limitações;
* Possíveis trabalhos futuros.

## 10. ENTREGÁVEIS
Cada grupo deverá entregar:
* Código-fonte do projeto ou notebook com os experimentos;
* Relatório em PDF;
* Arquivo `.csv` com as predições do melhor modelo.

## 11. REGRAS
* O arquivo auxiliar fornecido pelo professor não deve ser modificado.
* Todos os grupos devem usar a função `get_dataloaders` fornecida no arquivo auxiliar para carregar treino, validação e teste.
* O conjunto de teste deve ser usado apenas para avaliação final.
* O DataLoader do conjunto de teste deve usar `shuffle=False`, conforme definido no arquivo auxiliar.
* O arquivo final de resultados deve ser gerado usando a função `save_predictions`.
* O arquivo final deve se chamar `NOME1_NOME2_RESULTS.csv`.
* O arquivo final deve conter uma linha para cada imagem do conjunto de teste.
* As classes previstas devem ser representadas por números inteiros de 0 a 42.
* O arquivo de resultados deve conter obrigatoriamente as colunas `image_index` e `predicted_class`.
* O grupo deve garantir que os resultados apresentados no relatório correspondam ao mesmo modelo usado para gerar o arquivo `NOME1_NOME2_RESULTS.csv`.