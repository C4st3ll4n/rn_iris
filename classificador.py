import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


base = pd.read_csv('iris.csv')
previssores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
# setosa = 1 0 0 , virginica = 0 1 0, versicolor = 0 0 1
lcoder = LabelEncoder()
classe = lcoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

p_treinamento, p_teste, c_treinamento, c_teste = train_test_split(previssores, classe_dummy, test_size=0.25)


classificador = Sequential()

classificador.add(Dense(units = 4, input_dim=4, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation='softmax'))

classificador.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(p_treinamento, c_treinamento, batch_size=10, epochs=1000)

resultado = classificador.evaluate(p_teste,c_teste)
previsoes = classificador.predict(p_teste)

c_teste2 = [np.argmax(t) for t in c_teste ]
previsoes2 = [np.argmax(t) for t in previsoes ]
matriz = confusion_matrix(previsoes2, c_teste2)