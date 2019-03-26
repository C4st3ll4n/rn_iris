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

classificador = Sequential()
classificador.add(Dense(units = 8, input_dim=4, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation='softmax'))

classificador.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])


c_teste2 = [np.argmax(t) for t in classe_dummy ]
#previsoes2 = [np.argmax(t) for t in previsoes ]

classificador.fit(previssores, classe, batch_size=30, epochs=1000)


classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_iris.h5')