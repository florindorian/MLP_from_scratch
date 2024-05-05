import matplotlib.pyplot as plt
import numpy as np
from models.MlpModel import MlpModel


# -- Função para avaliar a eficácia em relação ao conjunto de Teste
def accuracy_test(x_test: np.ndarray, y_test: np.ndarray, model: MlpModel):
	length = y_test.shape[0]

	softmax = model.forward(x_test)
	predicted = np.zeros([length], dtype=int)
	for i in range(length):
		predicted[i] = softmax[i].argmax()

	comparation = np.zeros([length])
	for i in range(length):
		comparation[i] = predicted[i] == y_test[i]

	print(f'Accuracy: {comparation.sum() / length * 100:.2f}%')


# -- Função para plotar os gráficos da acurácia e da perda na comparação entre as abordagens:
# -- Variáveis Nomarlizadas X Variáveis Não Normalizadas
def plot_fit_comparation(fit_accuracy, fit_loss, fit_accuracyN, fit_lossN, epochs):
	plt.figure(figsize=(20, 5))  # -- Tamanho da figura
	plt.grid(True, linestyle='--', color='gray', linewidth=0.5)  # -- Grade
	plt.xticks(list(range(0, epochs + 1, 100)), rotation='vertical')  # -- Graduação do eixo x
	plt.yticks([i / 100 for i in range(0, 101, 5)])  # -- Graduação do eixo y
	plt.plot(list(range(1, epochs + 1)), fit_accuracy, 'lightgreen', label='Accuracy without Normalization')
	plt.plot(list(range(1, epochs + 1)), fit_loss, 'orange', label='Loss without Normalization')
	plt.plot(list(range(1, epochs + 1)), fit_accuracyN, 'lime', label='Accuracy with Normalization')
	plt.plot(list(range(1, epochs + 1)), fit_lossN, 'red', label='Loss with Normalization')
	plt.title('Performance de Treinamento: Variáveis Não Normalizadas X Variáveis Normalizadas', fontsize='20')
	plt.legend(loc='center', bbox_to_anchor=(0.5, 0.3))  # -- Legenda
	# -- Posição: bbox_to_anchor=(1,0.5) --> 0.5: Deslocamento Horizontal (% eixo X), 0.3: Deslocamento Vertical (% eixo Y)
	#             loc='upper right'      --> O centro do quadro da Legenda estará alinhado com o centro relativo anterior

	# plt.show()
