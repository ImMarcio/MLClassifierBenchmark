import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter  

# 📥  Carregar o dataset Titanic
df = pd.read_csv("datasets/titanic/TitanicDeathPrediction.csv")

#Deixando apenas as colunas relevantes e retirando linhas com valores nulos
df = df[['Age','Pclass','Sex','Fare','Survived']].dropna()

#Normalinando a coluna sex
column_transformer = make_column_transformer((OneHotEncoder(), ['Sex']), remainder='passthrough')
df = column_transformer.fit_transform(df)

#Removendo prefixos da transformação
columns_names = [
    name.split("__")[-1] for name in column_transformer.get_feature_names_out()
]

df = pd.DataFrame(data=df, columns=columns_names)



#Normalizando os dados numéricos 
df[['Fare', 'Age', 'Pclass']] = MinMaxScaler().fit_transform(df[['Fare', 'Age', 'Pclass']])


#Separando y e X
y = df.iloc[:, -1]  # Acessa a coluna na posição 6
X = df.iloc[:, :-1]  # Remove a coluna na posição 6

# Modelos de Machine Learning
tree_gini = DecisionTreeClassifier(criterion="gini")
tree_entropy = DecisionTreeClassifier(criterion="entropy")
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_10 = KNeighborsClassifier(n_neighbors=10)
mlp_relu = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", max_iter=2000, random_state=42)
mlp_tanh = MLPClassifier(hidden_layer_sizes=(100,), activation="tanh", max_iter=2000, random_state=42)
mlp_relu_large = MLPClassifier(hidden_layer_sizes=(200, 100), activation="relu", max_iter=2000, random_state=42)
mlp_tanh_large = MLPClassifier(hidden_layer_sizes=(200, 100), activation="tanh", max_iter=2000, random_state=42)
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)



# Dicionário com os modelos
models = {
    "Tree (Gini)": tree_gini,
    "Tree (Entropy)": tree_entropy,
    "kNN (k=5)": knn_5,
    "kNN (k=10)": knn_10,
    "MLP (ReLU)": mlp_relu,
    "MLP (Tanh)": mlp_tanh,
    "MLP Large (ReLu)": mlp_relu_large,
    "MLP Large (Tanh)": mlp_tanh_large,
    "K-Means": kmeans
}


# Variáveis para salvar a curva de erro das MLPs
loss_curve_relu = []
loss_curve_tanh = []
loss_curve_tanh_large = []
loss_curve_relu_large = []

#Números de interações 
folds = 10

kf = StratifiedKFold(n_splits = folds)


results = {}

for name, model in models.items():
    accuracies = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if "K-Means" in name:
            model.fit(X_train)
            cluster_labels_train = model.labels_
            cluster_labels_test = model.predict(X_test)

            mapping = {}

            for cluster_id in range(len(np.unique(y))):
                most_frequent_label = Counter(y_train[cluster_labels_train == cluster_id]).most_common(1)[0][0]
                mapping[cluster_id] = most_frequent_label

            y_pred = np.array([mapping[cluster_id] for cluster_id in cluster_labels_test])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        accuracies.append(acc)

        # Salvar a curva de erro para as MLPs
        if name == "MLP (ReLU)":
            loss_curve_relu.append(model.loss_curve_)
        elif name == "MLP (Tanh)":
            loss_curve_tanh.append(model.loss_curve_)
        elif name == "MLP Large (Tanh)":
            loss_curve_tanh_large.append(model.loss_curve_)
        elif name == "MLP Large (ReLu)":
            loss_curve_relu_large.append(model.loss_curve_)

    # Média dos 10 folds
    results[name] = np.mean(accuracies) * 100

# 📊  Exibir os resultados
df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy (%)"])
df_results = df_results.round(2)
print(df_results)


# Plotar as curvas de erro para MLP (ReLU) e MLP (Tanh)
for curve in loss_curve_relu:
    plt.plot(curve, label="MLP (ReLU) - Folds")
for curve in loss_curve_tanh:
    plt.plot(curve, label="MLP (Tanh) - Folds")
for curve in loss_curve_tanh_large:
    plt.plot(curve, label="MLP Large (Tanh) - Folds")
for curve in loss_curve_relu_large:
    plt.plot(curve, label="MLP Large (ReLu) - Folds")

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