import numpy as np
from PyQt5 import QtTest
import matplotlib.pyplot as plt
from graficador import Graficafor
from PyQt5 import QtGui


class Perceptron:

    # Constructor toma numero de inputs y learning rate
    def __init__(self, n_input, learning_rate):

        # inicializamos los pesos "w" con un vector de
        # dimension "n_input" con rango [-1,1] random
        self.w = -1 + (1 - (-1)) * np.random.rand(n_input)
        # bias random con rango [-1,1]
        self.b = -1 + (1 - (-1)) * np.random.rand()
        self.eta = learning_rate

        self.graficador = Graficafor()

    # funcion de activacion
    def f_activacion(self, num):
        if num >= 0:
            return 1
        else:
            return 0

    # Entrega un vector de salidas, dada una matriz de
    # entradas para la neurona

    def predict(self, X):

        # p es el numero de columnas en la matriz X
        p = X.shape[1]

        # y_est guardará la salidas, se inicializa
        # como vector de p dimensiones con 0s
        y_est = np.zeros(p)

        # iteramos por cada solucion a generar
        for i in range(p):
            # calculamos salidas
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            # asignamos valor binario según la funcion de activacion
            y_est[i] = self.f_activacion(y_est[i])

        # retornamos vector con las salidas binarias
        return y_est

    # Realiza aprendizaje en epocas

    def fit(self, X, Y, ui, epoch=50):
        # p es el numero de conjuntos de entrada (patron)
        p = X.shape[1]

        # iteramos por cada epoca
        for _ in range(epoch):
            # iteramos por cada patron
            for i in range(p):
                # calculamos salida dado el patron actual
                # reshape para asegurar que tenemos vector columna
                y_est = self.predict(X[:, i].reshape(-1, 1))

                # actualizacion de peso y bias basado en el error
                self.w = self.w + self.eta * (Y[i] - y_est) * X[:, i]
                self.b = self.b + self.eta * (Y[i] - y_est)

                # actualizacion en UI
                ui.txtW1.setText(str(round(self.w[0],4)))
                ui.txtW2.setText(str(round(self.w[1],4)))
                ui.txtTheta.setText(str(-round(self.b[0],4)))

                # plottear



            # retraso para visualizar
            QtTest.QTest.qWait(2000)


    # calcular punto
    def punto(self, w1, w2, teta, x):
        if w2 == 0:
            print("No se puede dividir entre cero, cambia el valor de W2")
            return
        
        m = -1*(w1/w2)
        c = teta/w2
        y = (m*x) + c
        return y

    def calcularPendiente(self,columna):
        # linea para dividir
        limLinea = [-10, 10]
        # print(self.w)
        # print(w[2])
        p1 = self.punto(self.w[0,columna], self.w[1,columna], -self.b, limLinea[0]), 
        p2 = self.punto(self.w[0,columna], self.w[1,columna], -self.b, limLinea[0]), 
        # p1 = self.punto(w1, w2, -self.b, limLinea[0]), 
        # p2 = self.punto(w1, w2, -self.b, limLinea[1])

        return p1,p2

    def guardarActualizar(self, ui):
        plt.savefig("prueba.png")
        ui.label.setPixmap(QtGui.QPixmap("prueba.png"))