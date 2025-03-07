# Comparação de algorítimos de Machine Learning 📊

**Objetivo:** Comparar os resultados de algoritmos de Machine Learning com problemas de classificação.

## Datasets utilizados 📂

1. [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) 🚢
2. [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) 💉

## Algoritmos de ML utilizados 🤖

* Árvore de Decisão (gini e entropy)
* kNN (k igual a 5 e 10)
* MLP (escolher duas arquiteturas diferentes e variar o parâmetro activation = {‘relu’,’tanh’}). “relu” é o valor default para o parâmetro activation
* K-Means (K igual ao número de classes existente no problema)

    Total: **9 algoritmos**

## Resultados dos treinamentos 📈

*Porcentagens em média das predições*

### Dataset - **Titanic** 🚢

Modelo             |      Acurácia
------------------ | ------------------
Tree (Gini)        |      77.75%
Tree (Entropy)     |      77.60%
kNN (k=5)          |      81.38%
kNN (k=10)         |      80.68%
MLP (ReLU)         |      79.14%
MLP (Tanh)         |      79.27%
MLP Large (ReLu)   |      81.24%
MLP Large (Tanh)   |      78.57%
K-Means            |      78.00%

### Dataset - **Diabetes** 💉

Modelo             |      Acurácia
------------------ | ------------------
Tree (Gini)        |      93.33
Tree (Entropy)     |      93.33
kNN (k=5)          |      89.33
kNN (k=10)         |      88.67
MLP (ReLU)         |      94.00
MLP (Tanh)         |      94.67
MLP Large (ReLu)   |      92.00
MLP Large (Tanh)   |      94.00
K-Means            |      88.

📌 *Este estudo proporciona insights valiosos sobre o desempenho de diferentes algoritmos em contextos distintos de classificação!*