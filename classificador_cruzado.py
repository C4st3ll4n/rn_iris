import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previssores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
lcoder = LabelEncoder()
classe = lcoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    
    classificador.add(Dense(units = 4, input_dim=4, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 4, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation='softmax'))
    
    classificador.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn=criar_rede, epochs=1000, batch_size=10)

resultado = cross_val_score(estimator= classificador,
                            X = previssores, y= classe, cv=10, scoring='accuracy')

media = resultado.mean()
desvio = resultado.std()