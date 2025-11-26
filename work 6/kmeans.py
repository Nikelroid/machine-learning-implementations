import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    first = generator.randint(0, n)
    centers = [first]
    for _ in range(1, n_cluster):
        dist_sq = np.array([min([np.sum((x[c] - x[i])**2) for c in centers]) for i in range(n)])
        if np.sum(dist_sq) == 0:
            non_centers = list(set(range(n)) - set(centers))
            if non_centers:
                centers.append(generator.choice(non_centers))
            else:
                centers.append(generator.randint(0, n))
        else:
            probs = dist_sq / np.sum(dist_sq)
            cumulative_probs = np.cumsum(probs)
            r = generator.rand()
            for j in range(n):
                if r <= cumulative_probs[j] and j not in centers:
                    centers.append(j)
                    break
            else:
                non_centers = list(set(range(n)) - set(centers))
                if non_centers:centers.append(generator.choice(non_centers))
                else:centers.append(generator.randint(0, n))
    return centers

def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

class KMeans():
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        centroids = x[self.centers]
        membership = np.zeros(N, dtype=int)
        iterations = 0
        for i in range(self.max_iter):
            iterations += 1
            distances = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                distances[:, k] = np.linalg.norm(x - centroids[k], axis=1)
            new_membership = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.n_cluster, D))
            for k in range(self.n_cluster):
                if np.sum(new_membership == k) > 0:new_centroids[k] = np.mean(x[new_membership == k], axis=0)
                else:new_centroids[k] = centroids[k]   
            if np.linalg.norm(new_centroids - centroids) < self.e:
                break     
            centroids = new_centroids
            membership = new_membership
        return centroids, membership, iterations
        
class KMeansClassifier():
    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"
        self.generator.seed(42)
        N, D = x.shape
        model = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, iterations = model.fit(x, centroid_func)
        centroid_labels = np.zeros((self.n_cluster,), dtype=int)
        for i in range(self.n_cluster):
            if np.sum(membership == i) > 0: 
                counts = np.bincount(y[membership == i])
                if len(counts) > 0:  
                    centroid_labels[i] = np.argmax(counts)
        self.centroid_labels = centroid_labels
        self.centroids = centroids
        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)
        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
            assert len(x.shape) == 2, "x should be a 2-D numpy array"
            N, D = x.shape
            distances = np.zeros((N, self.n_cluster)) 
            for i in range(self.n_cluster):
                distances[:, i] = np.sum((x - self.centroids[i])**2, axis=1)
            closest_cluster = np.argmin(distances, axis=1)
            predicted_labels = np.zeros(N, dtype=int)
            for i in range(N):
                predicted_labels[i] = self.centroid_labels[closest_cluster[i]]
            return predicted_labels

def transform_image(image, code_vectors):
    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'
    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    h, w, c = image.shape
    image_vector = image.reshape(-1, 3)
    model = KMeansClassifier(n_cluster=code_vectors.shape[0])
    dummy_labels = np.zeros(code_vectors.shape[0], dtype=int)
    model.fit(code_vectors, dummy_labels)
    model.centroids = code_vectors
    predictions = model.predict(image_vector)
    compressed_image = model.centroids[predictions].reshape(h, w, 3)
    return compressed_image
    
