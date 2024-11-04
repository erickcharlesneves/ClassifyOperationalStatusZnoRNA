# Classificação do estado operacional de Para-raios de ZnO

Projeto de inteligência artificial e ciências de dados realizado na graduação em Engenharia elétrica no LAT(Laboratório de Alta Tensão) da UFCG.

Este projeto tem como objetivo desenvolver e testar um classificador neural capaz de determinar o estado operacional de para-raios de ZnO (bom ou defeituoso) a partir da análise de sinais de corrente de fuga medidos em laboratório.

## Análise de Corrente de Fuga

## Objetivos Principais

- **Diagnosticar o estado operacional do para-raios**: classificando-o como `bom` ou `defeituoso`.
- **Prever falhas futuras**: garantindo a confiabilidade do sistema elétrico.

O uso de **redes neurais artificiais (RNA)** surge como uma abordagem eficaz para lidar com a complexidade dos dados de corrente de fuga fornecidos em testes de laboratório, caracterizados por sua alta não linearidade e variabilidade.

## ✨Dependências tecnológicas utilizadas:✨


### Instale as dependências das bibliotecas no seu ambiente python, são bibliotecas essenciais para análise de dados e machine learning, abrangendo desde a preparação dos dados até a modelagem, avaliação e visualização dos resultados, siga os passos de instalação e importação.

### 1. **Pandas**
   - Biblioteca poderosa para manipulação e análise de dados em Python. Ela facilita o trabalho com dados estruturados, como tabelas (DataFrames), e oferece funções para leitura, processamento e análise de dados.
   - **Principais usos**:
     - Manipular tabelas de dados (DataFrames).
     - Realizar operações de agregação, filtragem e limpeza de dados.
     - Ler e gravar arquivos CSV, Excel, SQL, JSON, entre outros formatos.
   - **Instalação**:
     ```bash
     pip install pandas
     ```
   - **Importação**:
     ```python
     import pandas as pd
     ```

### 2. **NumPy**
   - Biblioteca fundamental para computação científica em Python. Oferece suporte a arrays multidimensionais (ndarrays), operações matemáticas eficientes e funções para álgebra linear, transformadas de Fourier e geração de números aleatórios.
   - **Principais usos**:
     - Trabalhar com arrays e operações vetorizadas, mais eficientes que loops convencionais em Python.
     - Realizar operações matemáticas complexas e processamento de grandes volumes de dados numéricos.
   - **Instalação**:
     ```bash
     pip install numpy
     ```
   - **Importação**:
     ```python
     import numpy as np
     ```

### 3. **Scikit-Learn (sklearn)**
   - Biblioteca de machine learning em Python que inclui algoritmos de classificação, regressão, agrupamento, e ferramentas para pré-processamento e avaliação de modelos.
   - **Principais usos**:
     - **Divisão de dados**: `train_test_split` divide conjuntos de dados em treino e teste.
     - **Pré-processamento**: `StandardScaler` normaliza dados (escala média = 0, desvio padrão = 1).
     - **Modelagem**: `MLPClassifier` é uma rede neural perceptron multicamadas para classificação.
     - **Métricas de avaliação**: `accuracy_score`, `classification_report`, `confusion_matrix` e `roc_curve` calculam e avaliam o desempenho do modelo.
   - **Instalação**:
     ```bash
     pip install scikit-learn
     ```
   - **Importação**:
     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler
     from sklearn.neural_network import MLPClassifier
     from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
     ```

### 4. **Matplotlib**
   - Biblioteca de visualização que cria gráficos estáticos, animados e interativos. `matplotlib.pyplot` é uma interface simplificada para criar gráficos 2D.
   - **Principais usos**:
     - Criar gráficos de linha, barras, dispersão, histogramas e muito mais.
     - Customizar gráficos com legendas, rótulos de eixos e títulos.
   - **Instalação**:
     ```bash
     pip install matplotlib
     ```
   - **Importação**:
     ```python
     import matplotlib.pyplot as plt
     ```

### 5. **Seaborn**
   - Biblioteca de visualização de dados baseada no Matplotlib. Simplifica a criação de gráficos complexos e estiliza automaticamente os gráficos para torná-los mais atraentes e informativos.
   - **Principais usos**:
     - Visualizações de dados estatísticos, como gráficos de dispersão com linha de regressão, gráficos de caixa, gráficos de violino e matrizes de correlação.
     - Plotagem com temas prontos para visualizações claras e estéticas.
   - **Instalação**:
     ```bash
     pip install seaborn
     ```
   - **Importação**:
     ```python
     import seaborn as sns
     ```

#### Defina os caminhos para carregamento dos arquivos CSV de corrente de fuga.


## Metodologia de Desenvolvimento
### O processo de desenvolvimento seguiu as seguintes etapas:

1. **Aquisição e Processamento dos Dados**:
   - Foram fornecidos dois arquivos CSV (`corrente_PR_BOM.csv` e `corrente_PR_DEF.csv`) na pasta [`dados`](https://github.com/erickcharlesneves/ClassifyOperationalStatusZnoRNA/tree/main/dados), contendo sinais de corrente de fuga: um referente a para-raios em bom estado e outro a para-raios defeituosos.
   - Cada arquivo contém múltiplos sinais de corrente ao longo do tempo.

2. **Tratamento Inicial no [algoritmo](https://github.com/erickcharlesneves/ClassifyOperationalStatusZnoRNA/blob/main/src/Class_Neural_ParaRaioZno.py) em python**:
   - Remoção das primeiras 10 amostras para mitigar possíveis erros de aquisição.
   - Subamostragem dos sinais para conter exatamente 200 pontos, conforme solicitado, garantindo uniformidade e integridade dos dados por meio da função `resample` da biblioteca **Scipy**.

3. **Criação dos Alvos**:
   - Atribuição de rótulos: `1` para sinais de para-raios "BOM" e `0` para sinais de para-raios "DEF".
   - Combinação dos dados processados em um único conjunto com seus respectivos alvos.

4. **Divisão dos Dados**:
   - O conjunto de dados foi dividido em 80% para treinamento e 20% para teste usando a função `train_test_split`, garantindo a separação correta para validação do modelo.

5. **Normalização dos Dados**:
   - Para garantir que os dados fossem processados na mesma escala, a normalização foi realizada utilizando **StandardScaler**, o que evita que características com maior amplitude dominem o treinamento da rede.

## Arquitetura das RNAs

A rede neural utilizada no projeto foi o **Multilayer Perceptron (MLP)**, um tipo de rede neural feedforward.

## Algoritmo de Treinamento

O treinamento da rede neural é feito através de um processo de **retropropagação do erro**, ajustando os pesos de forma a minimizar a diferença entre as saídas previstas e as saídas reais. O algoritmo de **Adam (Adaptive Moment Estimation)** foi utilizado como otimizador, pois combina as vantagens de métodos de adaptação de taxa de aprendizado e momentum, proporcionando uma convergência rápida e eficiente.

## Função de Custo e Critério de Parada

A função de custo utilizada foi a **entropia cruzada**, adequada para problemas de classificação binária. O treinamento foi configurado para parar automaticamente (**early stopping**) quando a função de perda não apresentasse melhorias após 10 épocas consecutivas, prevenindo o **sobreajuste** (overfitting).

## Normalização dos Dados

A normalização dos dados foi um passo crítico no desenvolvimento do classificador. A corrente de fuga é composta por valores que variam em magnitude, e a **normalização z-score** foi utilizada.

## Subamostragem de Sinais

Uma das tarefas importantes do pré-processamento foi garantir que os sinais tivessem um número fixo de amostras, permitindo uma representação consistente para a rede neural. A técnica utilizada foi a **subamostragem** através do método `resample` da biblioteca **Scipy**, o qual reduziu os sinais para 200 pontos, preservando a integridade das informações relevantes.

## Métricas de Avaliação e Análise dos Resultados

### Acurácia

A **acurácia** é a métrica principal de avaliação, representando a porcentagem de classificações corretas. No projeto, a acurácia reflete a proporção de para-raios corretamente identificados como "bom" ou "defeituoso".

![image](https://github.com/user-attachments/assets/58920e2a-1fa6-4e60-8209-8d2f6d8b37f6)



Onde:
- **VP**: Verdadeiros Positivos (defeituoso corretamente identificado)
- **VN**: Verdadeiros Negativos (bom corretamente identificado)
- **FP**: Falsos Positivos (bom identificado incorretamente como defeituoso)
- **FN**: Falsos Negativos (defeituoso identificado incorretamente como bom)

O classificador obteve **100% de acurácia** tanto no conjunto de treinamento quanto no conjunto de teste, o que indica que o modelo aprendeu perfeitamente os padrões dos sinais de corrente de fuga. 

No algoritmo em Python, exibimos o gráfico da acurácia para comprovar a eficácia do modelo.

![image](https://github.com/user-attachments/assets/68559a82-0b4f-4f20-919d-edce9f1eb15c)

# Relatório de Classificação

Inclui precisão, recall e F1-score para ambas as classes ("BOM" e "DEF").

### 1. Precisão (Precision)
A **precisão** é uma métrica que avalia a proporção de previsões corretas entre as instâncias que foram classificadas como positivas pelo modelo.

![image](https://github.com/user-attachments/assets/5548d57d-5efe-40d8-9e5d-818a35bc65f1)


A precisão é importante quando os falsos positivos (prever positivo, mas é negativo) devem ser evitados.

### 2. Recall (Sensibilidade ou Revocação)
O **recall** mede a proporção de verdadeiros positivos que o modelo conseguiu identificar em relação ao total de instâncias realmente positivas.

![image](https://github.com/user-attachments/assets/2d066b6d-afbd-4739-a07a-ddb806d2754b)


O recall é essencial quando os falsos negativos (prever negativo, mas é positivo) são mais custosos e podem gerar prejuízos.

### 3. F1-Score
O **F1-Score** é a média harmônica entre a Precisão e o Recall. Ele é útil quando há um desequilíbrio entre as classes e desejamos encontrar um equilíbrio entre essas duas métricas.

![image](https://github.com/user-attachments/assets/cb21af93-f2d4-4808-8011-1b611ca43b83)

O F1-Score é útil quando há um trade-off entre precisão e recall, e ambas são importantes.

Exibimos o relatório assim no console do Spyder:

<p align="center">
  <img src="https://github.com/user-attachments/assets/67b06455-392c-45a3-9376-41e479d1634d" alt="Console spyder">
</p>

<div align="center">
<table>
  <thead>
    <tr>
      <th width="20%">Classe</th>
      <th width="20%">Precisão</th>
      <th width="20%">Recall</th>
      <th width="20%">F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DEF</td>
      <td>100%</td>
      <td>100%</td>
      <td>100%</td>
    </tr>
    <tr>
      <td>BOM</td>
      <td>100%</td>
      <td>100%</td>
      <td>100%</td>
    </tr>
  </tbody>
</table>
</div>

Este resultado mostra que o classificador foi perfeito em identificar tanto para-raios em bom estado quanto defeituosos.

Visualizamos essas métricas também em um gráfico de barras gerado no algoritmo em Python:

![image](https://github.com/user-attachments/assets/b5961b55-3482-4d8c-8ad6-8d029c1bb384)

## Matriz de Confusão

A **matriz de confusão** fornece uma visão detalhada das classificações corretas e incorretas, permitindo observar possíveis erros.

- Todas as 7 amostras de para-raios defeituosos foram classificadas corretamente como "DEF".
- Todas as 9 amostras de para-raios bons foram classificadas corretamente como "BOM".

Visualizamos assim conforme gerado no algoritmo em Python:

![image](https://github.com/user-attachments/assets/107aad06-e7bf-4c15-904b-7da7e9fc7a1c)


## Curva ROC e AUC

A **Curva ROC** (Receiver Operating Characteristic) é uma ferramenta usada para avaliar o desempenho de classificadores binários ao longo de diferentes limiares de decisão. A curva é plotada em um gráfico onde:

- O eixo x representa a **Taxa de Falsos Positivos (TFP)**: proporção de exemplos negativos classificados incorretamente como positivos.

![image](https://github.com/user-attachments/assets/d50de753-4a72-486b-a112-0637faa20065)

- O eixo y representa a **Taxa de Verdadeiros Positivos (TVP)** ou recall/sensibilidade: proporção de exemplos positivos classificados corretamente como positivos.

![image](https://github.com/user-attachments/assets/125fbc84-6eb5-462d-bae9-ed403a4f0471)

Uma curva ROC ideal é aquela que se aproxima do canto superior esquerdo do gráfico, onde a taxa de verdadeiros positivos é alta (próxima de 1) e a taxa de falsos positivos é baixa (próxima de 0).

A curva ROC do projeto praticamente alcançou o canto superior esquerdo do gráfico, e a **AUC** foi igual a 1.0, mostrando que o classificador separou perfeitamente as duas classes ("bom" e "defeituoso").

Com nosso algoritmo em Python, podemos plotar a curva conforme imagem abaixo comprovando isso:

![image](https://github.com/user-attachments/assets/637fd3ed-abf0-4ae4-b5fc-93e553d9e38f)


### AUC (Área Sob a Curva ROC)
A **AUC (Área Sob a Curva ROC)** é um número entre 0 e 1 que resume o desempenho do classificador:
- **AUC = 1.0**: Indica um classificador perfeito, que separa perfeitamente as classes.
- **AUC = 0.5**: Indica um classificador que faz previsões aleatórias.
- **AUC < 0.5**: Sugere que o classificador está invertendo as classes, o que indica um desempenho pior que o aleatório.

Neste projeto, a AUC obtida foi 1.0, indicando que o classificador neural teve um desempenho perfeito, separando completamente os para-raios bons dos defeituosos (TVP = 1 e TFP = 0).

## Gráficos de Perda

O gráfico de perda visualiza o comportamento da função de custo (perda) ao longo das iterações do treinamento da rede neural. Ele nos ajuda a entender se o modelo está:

- **Convergindo**: A perda diminui consistentemente e se estabiliza em um valor mínimo após várias épocas.
- **Overfitting**: A perda de treinamento continua a diminuir, mas a perda de validação começa a aumentar, indicando que o modelo está se ajustando demais aos dados de treinamento e generalizando mal.
- **Subajustando (Underfitting)**: A perda se estabiliza em um valor alto, indicando que o modelo não está conseguindo aprender adequadamente os padrões dos dados.

Neste projeto, o gráfico de perda mostrou uma diminuição consistente da perda de treinamento, sem sinais de overfitting. A função de custo foi monitorada com o critério de parada antecipada (**early stopping**), interrompendo o treinamento quando não houve melhora na perda de validação após 10 épocas conforme padrão de early stopping. 

O modelo foi capaz de generalizar bem para o conjunto de teste, sem sinais de sobreajuste, como evidenciado pelo comportamento da perda de validação. 
E conforme podemos visualizar ao plotar graficamente em nosso algoritmo em Python:

![image](https://github.com/user-attachments/assets/87fb1a39-cc5a-4536-b5c9-5c1798f12780)


## Conclusão

O classificador neural desenvolvido utilizando um **Multilayer Perceptron (MLP)** foi eficaz em 100% dos casos, para a classificação de para-raios ZnO em bom estado ou defeituosos, com resultados perfeitos em termos de acurácia, precisão e área sob a curva ROC (AUC). O uso de técnicas adequadas de pré-processamento dos dados, como subamostragem e normalização, foi crucial para o sucesso do modelo.

Dado o desempenho observado, este sistema pode ser considerado uma solução eficaz para monitoramento e diagnóstico do estado de para-raios em tempo real, auxiliando na tomada de decisões de manutenção preventiva e aumentando a confiabilidade dos sistemas elétricos.

Podemos ainda expandir futuramente o modelo com outros conjuntos de dados para testar o modelo com mais amostras, representando diferentes condições ambientais e operacionais, para garantir a robustez em cenários variados além da corrente de fuga.

E ainda explorar outras arquiteturas de redes neurais, como redes convolucionais (CNNs) ou recorrentes (RNNs) ou ainda árvores de decisão, que podem capturar padrões temporais de outras maneiras.

## License

[MIT license](https://github.com/erickcharlesneves/ClassifyOperationalStatusZnoRNA/blob/main/LICENSE) 

## Contribuições:

Contribuições são sempre bem-vindas. Sinta-se à vontade para levantar novas questões, e abrir pull requests. Considere dar uma estrela e bifurcar este repositório!

Se você tiver alguma dúvida sobre, não hesite em entrar em contato comigo no Gmail: [erick.cassiano@ee.ufcg.edu.br](mailto:erick.cassiano@ee.ufcg.edu.br) ou abrir um problema no GitHub.

## Contributing:

Contributions are always welcomed. Feel free to raise new issues, file new PRs. Consider giving it a star and fork this repo!

If you have any question about this opinionated list, do not hesitate to contact me on Gmail: [erick.cassiano@ee.ufcg.edu.br](mailto:erick.cassiano@ee.ufcg.edu.br) or open an issue on GitHub.
