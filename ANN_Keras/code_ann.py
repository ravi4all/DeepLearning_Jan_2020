import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import  Sequential
from keras.layers import Dense

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

label = LabelEncoder()
X[:,1] = label.fit_transform(X[:,1])
X[:,2] = label.fit_transform(X[:,2])

onehot = OneHotEncoder(categorical_features=[1])
X = onehot.fit_transform(X).toarray()

sc = StandardScaler()
X = sc.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# print(x_train.shape)

classifier = Sequential()
# input_neurons = 12
# next_layer_neurons = 6
# init = uniform (init weights with a uniform distribution (0.0 - 0.05))
classifier.add(Dense(input_dim=12,output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=32,nb_epoch=100)

y_pred = classifier.predict(x_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print("Accuracy is",acc)
print(cm)