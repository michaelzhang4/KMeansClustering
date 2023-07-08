from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()  # load in digits dataset from sklearn

data = scale(digits.data)   # scales down the data to make it easier to work with

y = digits.target   # stores dependent variable labels in y

k = 10  # the amount of clusters to be classified

def bench_k_means(estimator, name, data):   #function that trains the classifier and outputs it's performance metrics to console
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, estimator.inertia_,
           metrics.homogeneity_score(y, estimator.labels_),
           metrics.completeness_score(y, estimator.labels_),
           metrics.v_measure_score(y, estimator.labels_),
           metrics.adjusted_rand_score(y, estimator.labels_),
           metrics.adjusted_mutual_info_score(y, estimator.labels_),
           metrics.silhouette_score(data, estimator.labels_,
                                    metric='euclidean')))

clf = KMeans(n_clusters=k, init='random', n_init=10)    # create KMeans classifier with the initial assignment of 10 centroids being random and rerun 10 times in total
bench_k_means(clf, "Classifier_scores:", data)   # calls function to benchmark the KMeans model
