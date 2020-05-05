import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

%matplotlib inline

dig = load_digits()
plt.gray()
plt.matshow(dig.images[1792])

onehot_target = pd.get_dummies(dig.target)

x_train, x_val, y_train, y_val = train_test_split(dig.data, 
                                                  onehot_target,
                                                  test_size=0.1,
                                                  random_state=20)

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam

model = Sequential()
model.add(Dense(128, input_dim = x_train.shape[1], activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(optimizer = Adadelta(), 
              loss='categorical_crossentropy',
              metrics = ['categorical_accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=64)

scores = model.evaluate(x_val, y_val)

print(scores)