import numpy as np


class MlpModel:
    """
    Classe Implementadora de uma rede neural do tipo Multi-layer Perceptron (Perceptron Multicamadas)
    """

    # -- Definição dos atributos da rede MLP
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons: int = 10, output_neurons: int = 2):
        """
        Parâmetros:
        - x (numpy.ndarray): Array M x E de observações das variáveis de entrada.
            As colunas representam as variáveis, e as linhas as respectivas observações (instâncias).

        - y (numpy.ndarray): Vetor M x 1 dos resultados esperados

        - hidden_neurons (int): número de neurônios (perceptrons) ocultos

        - output_neurons (int): número de neurônios de saída (classes)
        """
        np.random.seed(8)
        self.x = x  # -- [M x E]
        self.y = y  # -- [M x 1]
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.input_neurons = self.x.shape[1]

        # -- Inicialização dos pesos e bias
        # -- Dados dos neurônios da camada Oculta (1)
        self.W1 = np.random.randn(self.input_neurons, self.hidden_neurons) / np.sqrt(self.input_neurons)  # -- [E x O]
        self.B1 = np.zeros((1, self.hidden_neurons))  # -- [1 x O]
        self.z1 = 0  # -- [M x O]
        self.f1 = 0  # -- [M x O]

        # -- Dados dos neurônios da camada de Saída (2)
        self.W2 = np.random.randn(self.hidden_neurons, self.output_neurons) / np.sqrt(self.hidden_neurons)  # -- [O x S]
        self.B2 = np.zeros((1, self.output_neurons))  # -- [1 x S]
        # -- Modelo da rede
        self.model_dict = {'W1': self.W1, 'B1': self.B1, 'W2': self.W2, 'B2': self.B2}

        # -- Dados de treino
        self.fit_accuracy = []
        self.fit_loss = []

    # -- Método de Treino da rede MLP
    def fit(self, epochs: int, lr: float):
        """
        Parâmetros:
        - epochs (int): número de épocas ("Iterações") de treino
        - lr (float): Learning Rate (Taxa de Aprendizado)

        Retorno:
        - (numpy.ndarray): um array que contém as previsões para cada lista de valores de entrada (x)
        """

        prediction = None
        for epoch in range(epochs):
            outputs = self.forward(self.x)  # -- forward retorna a softmax (as probabilidades de cada classe)
            loss = self.loss(outputs)
            self.backpropagation(outputs, lr)

            prediction = np.argmax(outputs, axis=1)  # -- classe com maior probabilidade

            # -- Quantidade de valores corretos, comparando valores preditos e valores de referência
            correct = (prediction == self.y).sum()
            accuracy = correct / self.y.shape[0]

            if int((epoch + 1) % (epochs / 10)) == 0:
                # -- Imprime os resultados a cada [(total de épocas) / 10] iterações, então há sempre 10 prints
                print(f'Epoch: [{epoch + 1} / {epochs}] Accuracy: {accuracy:.3f} Loss: {loss.item():.4f}')  # -- Formatação da saída

            # -- Registro dos dados de treino
            self.fit_accuracy.append(accuracy)
            self.fit_loss.append(loss.item())

        return prediction  # -- Retorna os valores para os quais foi feita a previsão

    # -- Método da etapa de Feedforward
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parâmetros:
        - x (numpy.ndarray): Array M x E de observações das variáveis de entrada.
            As colunas representam as variáveis (features), e as linhas as respectivas observações (instâncias).

        Retorno:
        - (numpy.ndarray): o array "softmax" [M x S] de probabilidades obtidas para cada classe representada pelo
            respectivo neurônio de saída
        """

        # -- Equação da reta (1):           z1 = x * W1 + B1      [M x E] @ [E x O] + [1 x O] = [M x O]
        self.z1 = x.dot(self.W1) + self.B1

        # -- Função de ativação (1):        f1 = tanh(z1)         [M x O]
        self.f1 = np.tanh(self.z1)

        # -- Equação da reta (2):           z2 = f1 * W2 + B2     [M x O] @ [O x S] + [1 x S] = [M x S]
        z2 = self.f1.dot(self.W2) + self.B2

        # -- Softmax
        exp_values = np.exp(z2)
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax  # -- [M x S]

    # -- Método que implementa a Função de Perda (Cross Entropy):
    def loss(self, softmax):
        """
        Parâmetros:
        - softmax (numpy.ndarray): o array "softmax" [M x S] de probabilidades obtidas para cada classe representada
            pelo respectivo neurônio de saída

        Retorno:
        - (numpy.float64): o valor que representa a perda em relação à saída gerada
        """

        # -- Seleção das probabilidades correspondentes às classes corretas/esperadas
        predictions = np.zeros(self.y.shape[0])  # -- [M x 1]
        for i, correct_index in enumerate(self.y):
            predicted = softmax[i][
                correct_index]  # -- A função de perda só é calculada em relação à classe correta/esperada
            predictions[i] = predicted

        # -- Cross Entropy:             CE = - y * log(ÿ)   ; y = 1 (pois 1 representa a probabilidade de se estar
        # certo. Logo, sempre será o valor esperado)
        log_prob = -np.log(predictions)

        return log_prob.mean()  # -- numpy.float64

    # -- Método que implementa a Retropropagação:
    def backpropagation(self, softmax: np.ndarray, learning_rate: float) -> None:
        """
        Faz o ajuste dos pesos e bias somando uma parcela de ajuste a eles

        Parâmetros:
        - softmax (numpy.ndarray): o array "softmax" [M x S] de probabilidades obtidas para cada classe representada
            pelo respectivo neurônio de saída
        - learning_rate (float): Learning Rate (Taxa de Aprendizado)
        """

        delta2 = np.copy(softmax)  # -- [M x S]
        delta2[range(self.x.shape[0]), self.y] -= 1  # -- dE/dÿ * dÿ/dz2 = ( - y * log(ÿ) )' * ( softmax(z2) )' = ÿ - 1
        # -- z2 = w2 * f1 + b2
        # -- dW2: derivada do erro em relação a w2        # -- ( dE/dÿ * dÿ/dz2 ) * dz2/dw2 = (ÿ - 1) * f1
        dW2 = (self.f1.T).dot(delta2)  # -- [M x O]' @ [M x S] = [O x M] * [M x S] = [O x S]
        # -- dB2: derivada do erro em relação a b2
        dB2 = np.sum(delta2, axis=0, keepdims=True)  # -- [M x S] --> [1 x S]

        delta1 = delta2.dot(self.W2.T) * (1 - np.power(np.tanh(self.z1), 2))
        # -- [ ( dE/dÿ * dÿ/dz2 ) * dz2/f1 * df1/dz1 ] = delta2 * w2 * tanh'(z1)
        # -- [M x S] @ [S x O] * [M x O] = [M x O]
        # -- dW1: derivada do erro em relação a w1  # -- [ ( dE/dÿ * dÿ/dz2 ) * dz2/f1 * df1/dz1 ] * dz1/dw1 = delta1 * x
        dW1 = self.x.T.dot(delta1)  # -- [M x E]' @ [M x O] = [E x M] @ [M x O] = [E x O]
        # -- dB1: derivada do erro em relação a b1
        dB1 = np.sum(delta1, axis=0, keepdims=True)  # -- [E x O] --> [1 x O]

        # -- Atualização dos pesos e bias
        # -- w_{n+1} = w_{n} + n * dE/dw        (dE/dw = delta * y)
        # -- b_{n+1} = b_{n} + n * dE/db        (dE/db = delta)
        self.W1 += - learning_rate * dW1
        self.W2 += - learning_rate * dW2
        self.B1 += - learning_rate * dB1
        self.B2 += - learning_rate * dB2

    # -- Método a ser usado para obter a acurácia e a perda obtidas em cada época de treinamento
    def get_fit_performance(self):
        return self.fit_accuracy, self.fit_loss
