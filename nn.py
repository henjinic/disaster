import datetime
import json
import numpy as np
import os
import tensorflow as tf
from maps import Maps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class NeuralNetwork:

    def __init__(self, x, y):
        self._x, self._y = x, y

        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  input_shape=(self._x.shape[1],), activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self._y.shape[1], activation='softmax')
        ])

        self._model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    def train(self, epochs, seed):
        self._epochs = epochs
        self._seed = seed

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(self._x, self._y,
                                                            test_size=0.2, random_state=seed)

        # normalize
        self._scaler = MinMaxScaler()
        self._scaler.fit(x_train)
        normalized_x_train = self._scaler.transform(x_train)
        normalized_x_test = self._scaler.transform(x_test)

        # train
        self._history = self._model.fit(normalized_x_train, y_train, verbose=1,
                                  validation_split=0.2, epochs=epochs)
        self._loss, self._accuracy = self._model.evaluate(normalized_x_test, y_test)

    def save(self, dirpath):
        os.mkdir(dirpath)

        # save model
        self._model.save(f'{dirpath}/model')

        # save train history
        with open(f'{dirpath}/history.json', 'w') as f:
            json.dump({k: [float(x) for x in v]
                       for k, v in self._history.history.items()}, f, indent=4)

        # save metadata
        results = {}
        results['loss'], results['accuracy'] = self._loss, self._accuracy
        results['epochs'] = self._epochs
        results['seed'] = self._seed

        with open(f'{dirpath}/metadata.txt', 'w') as f:
            json.dump(results, f, indent=4)

    def predict(self, features):
        return self._model.predict(self._scaler.transform(features))


def main():
    maps = Maps()
    curtime = datetime.datetime.today().strftime('%y%m%d-%H%M%S')
    dirpath = f'results/{curtime}'

    labels = ['0', '12345']
    x, y = maps.get_trainset(0.4, 0.6, labels=labels)

    nn = NeuralNetwork(x, y)

    nn.train(epochs=5, seed=0)
    nn.save(dirpath)

    probs = nn.predict(maps.valid_features)

    for i, label in enumerate(labels):
        maps.save_prob_map(probs[:, i], f'{dirpath}/prob_map_{label}.asc')

    maps.save_prob_map(np.argmax(probs, axis=1), f'{dirpath}/label_map.asc')


if __name__ == '__main__':
    main()
