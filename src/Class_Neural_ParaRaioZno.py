# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 21:13:47 2024

@author: erick
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Definir os caminhos para carregamento dos arquivos CSV de corrente de fuga
bom_file_path = r'C:\Users\erick\Documents\PythonTEE\corrente_PR_BOM.csv'
def_file_path = r'C:\Users\erick\Documents\PythonTEE\corrente_PR_DEF.csv'

# Ler os arquivos CSV
df_bom = pd.read_csv(bom_file_path)
df_def = pd.read_csv(def_file_path)

# remover os primeiros 10 valores (suposto erro de aquisição) e subamostrar para 200 amostras
from scipy.signal import resample
target_samples = 200
df_bom_clean = df_bom.iloc[:, 10:]
df_def_clean = df_def.iloc[:, 10:]
df_bom_resampled = pd.DataFrame(resample(df_bom_clean, target_samples, axis=1))
df_def_resampled = pd.DataFrame(resample(df_def_clean, target_samples, axis=1))

# Criar os alvos (labels)
y_bom = np.ones(len(df_bom_resampled))
y_def = np.zeros(len(df_def_resampled))

# Combinar os dados
X = pd.concat([df_bom_resampled, df_def_resampled], axis=0)
y = np.concatenate([y_bom, y_def])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #semente aleatoria reproduzivel 42 seja a mesma

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementar o classificador neural com monitoramento da perda
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, verbose=True, early_stopping=True)
mlp.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = mlp.predict(X_test_scaled)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Exibir gráfico da acurácia
plt.figure(figsize=(6,4))
plt.bar(["Acurácia"], [accuracy * 100], color='skyblue')
plt.ylim(0, 100)
plt.title('Acurácia do Classificador')
plt.ylabel('Acurácia (%)')
plt.show()

# Gerando o relatório de classificação para visu em barras
report = classification_report(y_test, y_pred, target_names=["Bom", "Defeituoso"])

# Exibindo o relatório
print("Relatório de Classificação:\n", report)

# Dados do relatório de classificação
report_dict = classification_report(y_test, y_pred, target_names=["Bom", "Defeituoso"], output_dict=True)

# Extraindo métricas de precisão, recall e F1-score
labels = ["Bom", "Defeituoso"]
metrics = ["precision", "recall", "f1-score"]
values = [[report_dict[label][metric] for metric in metrics] for label in labels]

# Convertendo para numpy array para manipulação
values = np.array(values)

# Plotando gráfico de barras
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(labels))
width = 0.2

# Plotando as métricas
ax.bar(x - width, values[:, 0], width, label="Precisão")
ax.bar(x, values[:, 1], width, label="Revocação")
ax.bar(x + width, values[:, 2], width, label="F1-Score")

# Labels e títulos
ax.set_ylabel("Pontuação")
ax.set_title("Precisão, Revocação e F1-Score por classe")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Mostrando o gráfico de classif
plt.show()

# Matriz de confusão para analisar classificação
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Defeituoso", "Bom"], yticklabels=["Defeituoso", "Bom"])
plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.show()

# Plotar a curva ROC
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (TFP)')
plt.ylabel('Taxa de Verdadeiros Positivos (TVP)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Plotar gráfico de perda de treinamento
plt.figure(figsize=(6,4))
plt.plot(mlp.loss_curve_, label='Perda de Treinamento')
plt.title('Perda de Treinamento ao Longo das Épocas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()
