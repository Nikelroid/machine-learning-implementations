import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
        KNN.features = features
        KNN.labels = labels

    def get_k_neighbors(self, point):
        distances = []
        for ind,(fea,lab) in enumerate(zip(KNN.features,KNN.labels)):
            distances.append([self.distance_function(point, fea),lab,ind])
        distances.sort(key=lambda x: (x[0],x[2]))
        distances = distances[:self.k]
        return [int(distances[i][1]) for i in range(len(distances))]
		
    def predict(self, features):
        counts = []
        for feature in features:
            neighbors = self.get_k_neighbors(feature)
            counts.append(int(Counter(neighbors).most_common(1)[0][0]))
        return [counts][0]


if __name__ == '__main__':
    print(np.__version__)
