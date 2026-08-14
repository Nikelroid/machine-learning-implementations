import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    TP = 0
    F = 0
    for predicted_label, real_label in zip(predicted_labels, real_labels):
        if real_label==1 and predicted_label==1:
            TP += 1
        elif real_label != predicted_label:
            F += 1
    
    return (2*TP)/(2*TP+F)
    

class Distances:
    @staticmethod
    
    def minkowski_distance(point1, point2):
        d3_sum= 0
        for p1, p2 in zip(point1, point2):
            d3_sum += abs(p1-p2)**3
        return np.cbrt(d3_sum)

    @staticmethod
    def euclidean_distance(point1, point2):
        d2_sum= 0
        for p1, p2 in zip(point1, point2):
            d2_sum += (p1-p2)**2
        return np.sqrt(d2_sum)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        if np.linalg.norm(point1)==0 or np.linalg.norm(point2)==0:
            return 1
        else:
            return 1 - np.dot(point1,point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        f1_scores = []
        for k in ks:
            for func_index, funcs in enumerate(distance_funcs.values()):
                model = KNN(k, funcs)
                model.train(x_train, y_train)
                y_val_pred = model.predict(x_val)
                f1_scores.append([f1_score(y_val, y_val_pred),np.max(ks)-k,len(distance_funcs)-func_index])
        f1_scores.sort(key=lambda x: (x[0],x[2],x[1]), reverse=True)
        self.best_k = np.max(ks)-f1_scores[0][1]
        self.best_distance_function = list(distance_funcs.keys())[len(distance_funcs)-f1_scores[0][2]]
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        pass

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        f1_scores = []
        for scaler_index , scaler in enumerate(scaling_classes.values()):
            transformer = scaler()
            for k in ks:
                for func_index, funcs in enumerate(distance_funcs.values()):
                    model = KNN(k, funcs)
                    model.train(transformer(x_train), y_train)
                    y_val_pred = model.predict(transformer(x_val))
                    f1_scores.append([f1_score(y_val, y_val_pred),np.max(ks)-k,len(distance_funcs)-func_index, len(scaling_classes)-scaler_index])
        f1_scores.sort(key=lambda x: (x[0],x[3],x[2],x[1]), reverse=True)
        self.best_k = np.max(ks)-f1_scores[0][1]
        self.best_distance_function = list(distance_funcs.keys())[len(distance_funcs)-f1_scores[0][2]]
        self.best_scaler = list(scaling_classes.keys())[len(scaling_classes)-f1_scores[0][3]]
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        pass


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        normalized_features = []
        for feature in features:
            norm = np.linalg.norm(feature)
            if np.sum(np.abs(feature)) != 0:
                normalized_features.append(feature/ norm)
            else:
                normalized_features.append(feature)
        return normalized_features 


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        features_transpose = np.array(features).T
        normalized = []
        for feature in features_transpose:
            minimum_row = np.min(feature)
            maximum_row = np.max(feature)
            if maximum_row != minimum_row:
                normalized.append((feature - minimum_row)/(maximum_row - minimum_row))
            else:
                normalized.append(np.zeros_like(features_transpose[0]))
        return np.array(normalized).T
    