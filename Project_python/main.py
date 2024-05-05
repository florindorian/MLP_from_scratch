# -- Importação das bibliotecas da linguagem
import pandas as pd
import sklearn.datasets as datasets
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -- Importação do modelo e funções auxiliares
from models.MlpModel import MlpModel
from functions.functions import plot_fit_comparation
from functions.functions import accuracy_test

# -- Configurações dos gráficos
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('dark_background')


# -- Importação do Dataset: "Breast cancer wisconsin (diagnostic) dataset"
df = datasets.load_breast_cancer(as_frame=True)

# -- Seleção dos dados
x = df.data.to_numpy().copy()[:, 0:10] # -- Variáveis sem normalização
xN = df.data.to_numpy().copy()[:, 0:10] # -- Variáveis a serem normalizadas
y = df.target.to_numpy().copy() # -- Saídas esperadas


# -- Normalização das variáveis em xN por Máximo e Mínimo
for i in list(range(xN.shape[1])):
	max_num = xN[:, i].max()
	min_num = xN[:, i].min()
	xN[:, i] = (xN[:, i] - min_num) / (max_num - min_num)


# -- Divisão dos dados em conjuntos de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=8)
xN_train, xN_test, yN_train, yN_test = train_test_split(xN, y, train_size=0.7, random_state=8)


# -- Configuração dos hiperparâmetros
hidden = 6  # -- hidden = 6
output = 2  # -- output = 2
epochs = 7700  # -- epochs = 7700
lr = 0.0013  # -- lr = 0.0013


# -- Treinamento com variáveis Não Normalizadas
print(f'\nTreinamento com variáveis Não Normalizadas')
cancerModel = MlpModel(x=x_train, y=y_train, hidden_neurons=hidden, output_neurons=output)
result = cancerModel.fit(epochs, lr)


# -- Treinamento com variáveis Normalizadas
print(f'\n\nTreinamento com variáveis Normalizadas')
cancerModelN = MlpModel(x=xN_train, y=yN_train, hidden_neurons=hidden, output_neurons=output)
resultN = cancerModelN.fit(epochs, lr)


# -- Acurácia em relação ao conjunto de Teste
print("\n\nAcurácia: Com Variáveis Não Normalizadas: ")
accuracy_test(x_test, y_test, cancerModel)

print("\nAcurácia: Com Variáveis Normalizadas")
accuracy_test(xN_test, yN_test, cancerModelN)


# -- Declaração das variáveis
opt = -1
while (opt != 0):
	print(f'\n\nO que você gostaria de fazer?')
	print(f'[0] Sair')
	print(f'[1] Plotar gráficos comparativos do treinamento e dos resultados')
	print(f'[2] Salvar o modelo em .pickle (com variáveis normalizadas)')
	opt = int(input(f'Opção escolhida: '))

	if (opt == 1):
		# -- Comparação entre os treinamentos com e sem Normalização das Variáveis
		fit_accuracy, fit_loss = cancerModel.get_fit_performance()
		fit_accuracyN, fit_lossN = cancerModelN.get_fit_performance()

		plot_fit_comparation(fit_accuracy, fit_loss, fit_accuracyN, fit_lossN, epochs)

		# -- Comparação entre os resultados com e sem Normalização das Variáveis
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
		fig.suptitle('Resultado das Classificações', fontsize=15)
		ax1.set_title('Variáveis não normalizadas', fontsize=10, y=1.0)
		ax2.set_title('Variáveis normalizadas', fontsize=10, y=1.0)

		confusion_matrix = mt.confusion_matrix(y_train, result)
		confusion_matrix = pd.DataFrame(confusion_matrix).rename_axis("Valor Predito", axis=1).rename_axis(
			"Valor Correto", axis=0)
		sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm', fmt='.0f', ax=ax1)

		confusion_matrixN = mt.confusion_matrix(yN_train, resultN)
		confusion_matrixN = pd.DataFrame(confusion_matrixN).rename_axis("Valor Predito", axis=1).rename_axis(
			"Valor Correto", axis=0)

		sns.heatmap(confusion_matrixN, annot=True, cmap='coolwarm', fmt='.0f', ax=ax2)

		plt.show()

	elif (opt == 2):
		with open('modelo_treinado.pickle', 'wb') as f:
			pickle.dump(cancerModelN, f)  # -- Salvar 'bc' dentro de f
