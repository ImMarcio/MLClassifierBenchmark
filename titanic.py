import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score



# 📥  Carregar o dataset Titanic
df = pd.read_csv("datasets/titanic/train.csv")

# Selecionar apenas colunas relevantes e tratar valores faltantes
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

# Converter 'Sex' para valores numéricos (0 = Male, 1 = Female)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Normalizar apenas as colunas numéricas (Age, Fare, Pclass)
scaler = MinMaxScaler()
df[['Age', 'Fare', 'Pclass']] = scaler.fit_transform(df[['Age', 'Fare', 'Pclass']])

# Exibir os primeiros dados
print(df.head())

# Preparar os dados para treinamento e teste
kf = KFold(n_splits=10, shuffle=True, random_state=42)
X = df.iloc[:, 1:].values  # Atributos
y = df.iloc[:, 0].values   # Labels

# Modelos de Machine Learning
tree_gini = DecisionTreeClassifier(criterion="gini")
tree_entropy = DecisionTreeClassifier(criterion="entropy")
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
mlp_relu = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=2000, random_state=42)
mlp_tanh = MLPClassifier(hidden_layer_sizes=(100,), activation="tanh", max_iter=2000, random_state=42)
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)

# Dicionário com os modelos
models = {
    "Tree (Gini)": tree_gini,
    "Tree (Entropy)": tree_entropy,
    "kNN (k=5)": knn_5,
    "kNN (k=10)": knn_10,
    "MLP (ReLU)": mlp_relu,
    "MLP (Tanh)": mlp_tanh,
    "K-Means": kmeans
}

# Rodar os Modelos e Calcular as Métricas
results = {}

# Variáveis para salvar a curva de erro das MLPs
loss_curve_relu = []
loss_curve_tanh = []

for name, model in models.items():
    accuracies = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Treinar e testar o modelo
        if "K-Means" in name:
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.array([y_train[i] for i in y_pred])  # Ajustar rótulos
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calcular acurácia
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Salvar a curva de erro para as MLPs
        if name == "MLP (ReLU)":
            loss_curve_relu.append(model.loss_curve_)
        elif name == "MLP (Tanh)":
            loss_curve_tanh.append(model.loss_curve_)
    
    # Média dos 10 folds
    results[name] = np.mean(accuracies) * 100

# 📊  Exibir os resultados
df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy (%)"])
print(df_results)

# Plotar as curvas de erro para MLP (ReLU) e MLP (Tanh)
# As curvas de erro podem ter comprimentos diferentes, então plotaremos todas as iterações separadamente

for curve in loss_curve_relu:
    plt.plot(curve, label="MLP (ReLU) - Folds")
for curve in loss_curve_tanh:
    plt.plot(curve, label="MLP (Tanh) - Folds")

plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.title("Evolução do Erro no Treinamento das MLPs")
plt.legend()
plt.show()


# Plotando o gráfico de barras com seaborn
plt.figure(figsize=(10, 5))  # Tamanho da figura
sns.barplot(x=df_results.index, y=df_results["Accuracy (%)"])  # Plotando a acurácia média por modelo
plt.xticks(rotation=45)  # Rotacionando os rótulos no eixo X
plt.title("Comparação de Algoritmos (Acurácia Média)")  # Título do gráfico
plt.ylabel("Acurácia (%)")  # Rótulo do eixo Y
plt.show()  # Exibir o gráfico
