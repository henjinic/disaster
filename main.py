import datetime
import numpy as np
from maps import Maps
from nn import NeuralNetwork


EPOCHS = 5

BOUNDS_ENTRIES = [
    [0.2, 0.8],
    [0.3, 0.7],
    [1.0/3, 2.0/3],
    [0.4, 0.6]
]

LABELS_ENTRIES = [
    ['0', '12345'],
    ['1', '02345'],
    ['2', '01345'],
    ['3', '01245'],
    ['4', '01235'],
    ['5', '01234'],
    ['2', '3', '4', '5'],
    ['1', '2', '3', '4', '5'],
    ['0', '1', '2', '3', '4', '5'],
]

def main():
    maps = Maps()

    for i, bounds in enumerate(BOUNDS_ENTRIES):
        for j, labels in enumerate(LABELS_ENTRIES):
            curtime = datetime.datetime.today().strftime('%y%m%d-%H%M%S')
            dirpath = f'results/{curtime}_{i}_{j}'

            x, y = maps.get_trainset(*bounds, labels=labels)

            nn = NeuralNetwork(x, y)

            nn.train(epochs=EPOCHS, seed=0)
            nn.save(dirpath)

            probs = nn.predict(maps.valid_features)

            for k, label in enumerate(labels):
                maps.save_prob_map(probs[:, k], f'{dirpath}/prob_map_{label}.asc')

            maps.save_prob_map(np.argmax(probs, axis=1), f'{dirpath}/label_map.asc')



if __name__ == '__main__':
    main()
