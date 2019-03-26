from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

arq = open('classificador_iris.json','r')
estrutura = arq.read()
arq.close()

classificador = model_from_json(estrutura)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[5.09, 3.5, 1.39, 2.0]])

previssao = classificador.predict(novo)
previssao_b = (previssao > 0.5)


base = pd.read_csv('iris.csv')
previssores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
# setosa = 1 0 0 , virginica = 0 1 0, versicolor = 0 0 1
lcoder = LabelEncoder()
classe = lcoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])

resultado = classificador.evaluate(previssores,classe)