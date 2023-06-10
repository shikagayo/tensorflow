import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

from keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words = 10000)

data = np.concatenate((training_data, testing_data), axis = 0)
targets = np.concatenate((training_targets, testing_targets), axis = 0)

# print('Категорії: ', np.unique(targets))
# print('Особливі слова: ', len(np.unique(np.hstack(data))))

# length = [len(i) for i in data]
# print(np.mean(length))
# print(round(np.std(length)))

# print('Текст: ', targets[0])

# print(data[0])

# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items])
# decoded = ' '.join([reverse_index.get(i-3, '#') for i in data[0]])
# print(decoded)

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

data = vectorize(data)
targets = np.array(targets).astype('float32')

test_x = data[:10000]
test_y = targets[:10000]

train_x = data[10000:]
train_y = targets[10000:]

model = models.Sequential()

model.add(layers.Dense(50,activation = 'relu', input_shape = (10000, )))

model.add(layers.Dropout(0.3, noise_shape = None, seed = None))
model.add(layers.Dense(50, activation = 'relu'))

model.add(layers.Dropout(0.2, noise_shape = None, seed = None))
model.add(layers.Dense(50, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

results = model.fit(
    train_x, train_y, epochs = 2,
    batch_size = 500,
    validation_data = (test_x, test_y)
)

print(np.mean(results.history['val_acc']))