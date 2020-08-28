import numpy as np


DISASTERS = [
    're_fire',
    're_slide',
    're_flood',
    're_heat_pop',
    're_rice',
]

FEATURES = [
    're_bio1',
    're_bio2',
    're_bio3',
    're_bio4',
    're_bio5',
    're_bio6',
    're_bio7',
    're_bio8',
    're_bio9',
    're_bio10',
    're_bio11',
    're_bio12',
    're_bio13',
    're_bio14',
    're_bio15',
    're_bio16',
    're_bio17',
    're_bio18',
    're_bio19',
    're_population',
    're_dem',
    're_geomorphon',
]


class Maps:

    def __init__(self):
        with open(f'disasters/{DISASTERS[0]}.asc') as f:
            self._header = f.readlines()[:6]

        self._ncols = int(self._header[0].split()[1])
        self._nrows = int(self._header[1].split()[1])

        disasters = self._load(DISASTERS, 'disasters')
        features = self._load(FEATURES, 'features')

        self._disasters = np.dstack(disasters).reshape(-1, len(disasters))
        self._features = np.dstack(features).reshape(-1, len(features))

        self._valid_feature_mask = (self._features != -9999).all(axis=1)

    @property
    def valid_features(self):
        result = self._features[self._valid_feature_mask]

        result = self._onehot_encode(result, 21, ref=list(range(1, 11)))

        return result

    def get_trainset(self, lower_bound, upper_bound, labels=['0', '12345']):
        x, y = [], []

        for feature_vec, disaster_vec in zip(self._features, self._disasters):
            if -9999 in feature_vec:
                continue

            if -9999 in disaster_vec:
                continue

            categories = []

            for disaster in disaster_vec:
                if disaster <= lower_bound:
                    categories.append(0)
                elif disaster >= upper_bound:
                    categories.append(1)
                else:
                    categories.append(2)

            if 2 in categories:
                continue

            for i, label in enumerate(labels):
                vec = [0] * len(labels)
                if str(sum(categories)) in label:
                    vec[i] = 1
                    y.append(vec)
                    x.append(feature_vec)
                    break

        return self._onehot_encode(np.array(x), 21, ref=list(range(1, 11))), np.array(y)

    def save_prob_map(self, probs, path):
        prob_map = np.zeros(self._nrows * self._ncols)
        prob_map[:] = -9999
        prob_map[self._valid_feature_mask] = probs
        prob_map = prob_map.reshape(self._nrows, self._ncols)

        with open(path, 'w') as f:
            f.write(''.join(self._header))
            for row in prob_map:
                f.write(' '.join(map(str, row)) + '\n')

    def _load(self, names, directory):
        maps = []

        for name in names:
            with open(f'{directory}/{name}.asc') as f:
                maps.append(np.loadtxt(f, skiprows=6).reshape(self._nrows, self._ncols))

        return maps

    def _onehot_encode(self, features, ith, ref=[]):
        target_column = features[:, ith]

        result = []

        if ref:
            for val in target_column:
                onehot_vec = [0.0] * (len(ref) - 1)

                if val != ref[-1]:
                    onehot_vec[int(val - 1)] = 1.0
                result.append(onehot_vec)
        else:
            uniques, labels = np.unique(target_column, return_inverse=True)

            for label in labels:
                onehot_vec = [0.0] * (len(uniques) - 1)

                if label != (len(uniques) - 1):
                    onehot_vec[label] = 1.0
                result.append(onehot_vec)

        return np.concatenate((features[:, :ith], result, features[:, ith+1:]), axis=1)


def main():
    maps = Maps()

    x, y = maps.get_trainset(0.4, 0.6, ['0', '1', '2', '3', '4', '5'])
    print(np.sum(y, axis=0))

    print(maps.valid_features)


if __name__ == '__main__':
    main()
