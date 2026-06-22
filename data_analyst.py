import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importações obrigatórias dos arquivos do seu projeto
from src.utils.gtsrb import GTSRB_CLASSES, get_dataloaders, save_predictions
from src.pipe.pipeline import BASE_TRANSFORM

RUNS_DIR = "results/runs"

def get_ground_truth_labels():
    """
    Obtém os rótulos reais do conjunto de teste utilizando a função obrigatória
    get_dataloaders para garantir o alinhamento exato com o DataLoader de teste.
    """
    print("Carregando o DataLoader de teste para extrair os rótulos reais (y_true)...")
    # get_dataloaders exige transform, batch_size e opcionalmente data_augmentation
    _, _, test_loader = get_dataloaders(
        transform=BASE_TRANSFORM,
        batch_size=64,
        data_augmentation=False
    )
    
    y_true = []
    for _, targets in test_loader:
        y_true.extend(targets.tolist())
        
    return np.array(y_true)

def plot_training_curves(history: dict, exp_name: str, save_path: str):
    """
    Gera e salva os gráficos das curvas de Loss e Acurácia solicitados no relatório.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Curvas de Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    ax1.set_title(f'Curva de Perda (Loss) - {exp_name}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Curvas de Acurácia
    ax2.plot(epochs, history['train_accuracy'], label='Train Acc', marker='o')
    ax2.plot(epochs, history['val_accuracy'], label='Val Acc', marker='o')
    ax2.set_title(f'Curva de Acurácia - {exp_name}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "curves_plot.png"))
    plt.close()

def compute_metrics_and_plots(y_true: np.ndarray, y_pred: np.ndarray, exp_name: str, save_path: str) -> dict:
    """
    Calcula as métricas globais, por classe e plota a Matriz de Confusão.
    """
    # Métricas globais exigidas
    global_acc = accuracy_score(y_true, y_pred)
    
    # Matriz de Confusão (Exigência do item 9.4 e 10)
    cm = confusion_matrix(y_true, y_pred)
    
    # Cálculo manual da Acurácia por Classe para extrair Macro-Acurácia de forma transparente
    # Evita divisões por zero se houver alguma classe sem amostras no test set
    class_accuracies = []
    for i in range(43):
        total_samples = cm[axis := 0 if False else i].sum()
        if total_samples > 0:
            acc = cm[i, i] / total_samples
        else:
            acc = 0.0
        class_accuracies.append(acc)
        
    macro_acc = np.mean(class_accuracies)
    
    # Mapeamento para identificar as classes mais difíceis (Menor acurácia)
    class_perf = {GTSRB_CLASSES[i]: class_accuracies[i] for i in range(43)}
    hardest_classes = sorted(class_perf.items(), key=lambda x: x[1])[:5]
    
    # Plotagem da Matriz de Confusão estruturada
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap='magma', cbar=True)
    plt.title(f'Matriz de Confusão - {exp_name}\nAcurácia Global: {global_acc:.4f} | Macro-Acurácia: {macro_acc:.4f}', fontsize=16)
    plt.xlabel('Predito (Classe)', fontsize=12)
    plt.ylabel('Verdadeiro (Classe)', fontsize=12)
    
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        "experiment": exp_name,
        "global_acc": global_acc,
        "macro_acc": macro_acc,
        "hardest_classes": hardest_classes,
        "raw_preds": y_pred
    }

def main():
    if not os.path.exists(RUNS_DIR):
        print(f"Diretório '{RUNS_DIR}' não encontrado. Execute seus experimentos primeiro.")
        return

    # Garante os alinhamentos dos labels reais conforme os loaders oficiais
    y_true = get_ground_truth_labels()
    
    experiments = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    results_summary = []
    
    print(f"\nProcessando {len(experiments)} experimentos encontrados...")
    
    for exp_name in experiments:
        exp_path = os.path.join(RUNS_DIR, exp_name)
        
        # 1. Carregar Histórico e gerar curvas
        history_path = os.path.join(exp_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            plot_training_curves(history, exp_name, exp_path)
            
        # 2. Carregar predições geradas pelo pipeline (.csv)
        preds_csv_path = os.path.join(exp_path, "predictions.csv")
        if os.path.exists(preds_csv_path):
            df_preds = pd.read_csv(preds_csv_path)
            # Ignora linhas comentadas se houver, garante ordenação pelo image_index
            df_preds = df_preds.sort_values(by="image_index")
            y_pred = df_preds["predicted_class"].values
            
            # Análise estatística e matriz
            metrics = compute_metrics_and_plots(y_true, y_pred, exp_name, exp_path)
            results_summary.append(metrics)
            
            print(f" -> Experimento '{exp_name}' processado com sucesso.")

    # 3. Determinação do melhor modelo e exportação final
    if results_summary:
        print("\n" + "="*70)
        print("RANKING GERAL DOS EXPERIMENTOS (Ordenado por Acurácia Global)")
        print("="*70)
        
        df_ranking = pd.DataFrame(results_summary).sort_values(by="global_acc", ascending=False)
        for idx, row in df_ranking.iterrows():
            print(f"Modelo: {row['experiment']:<30} | Acc Global: {row['global_acc']:.4f} | Macro Acc: {row['macro_acc']:.4f}")
            
        best_exp = df_ranking.iloc[0]
        print("="*70)
        print(f"🏆 MELHOR MODELO DETECTADO: {best_exp['experiment']}")
        print(f"Acurácia Global: {best_exp['global_acc']:.4f} | Macro-Acurácia: {best_exp['macro_acc']:.4f}")
        print("="*70)
        
        print("\nClasses com maior índice de dificuldade (Top 5 Piores):")
        for cls_name, acc in best_exp['hardest_classes']:
            print(f"  - {cls_name}: {acc:.2%} de acertos")
            
        # GERAÇÃO DO ENTREGÁVEL VIA FUNÇÃO OBRIGATÓRIA `save_predictions`
        final_output_name = "JoãoPedroSantana_NathanRezende_ThiagoFernandes_RESULTS.csv" 
        best_preds_tensor = torch.tensor(best_exp['raw_preds'])
        
        print(f"\nGerando arquivo de entrega final usando 'save_predictions'...")
        save_predictions(
            y_pred=best_preds_tensor,
            filepath=final_output_name,
            experiment_name=f"Melhor Modelo: {best_exp['experiment']}"
        )
        print(f"Sucesso! Arquivo '{final_output_name}' exportado perfeitamente pronto para submissão.")

if __name__ == "__main__":
    main()