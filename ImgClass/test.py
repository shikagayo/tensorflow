from image import load_data, batch_size
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

categories = {
    0: 'airplane',
    1: 'automodile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

ds_train, ds_test, info = load_data()
model = load_model('results/cifar10-model-v1.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy*100, '%')

data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])

print('Prediction label: ', categories[prediction])
print('True label: ', sample_label)