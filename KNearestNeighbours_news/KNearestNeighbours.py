import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
plt.style.use('ggplot')


'''
Load news articles from CSV files
'''
news1 = pd.read_csv('armenpress_army.csv', encoding='utf8')
news2 = pd.read_csv('armenpress_economy.csv', encoding='utf8')

'''
Preparing the data for the model
'''
news1.article_paragraph = news1.article_paragraph.str.split('[0-9], ARMENPRESS.', expand=True)[1]
news2.article_paragraph = news2.article_paragraph.str.split('[0-9], ARMENPRESS.', expand=True)[1]


news1['type'] = 'military'
news2['type'] = 'economy'

news_df = pd.concat([news1, news2], axis=0, ignore_index=True)
news_df = news_df.dropna()

X_train, X_test, y_train, y_test = train_test_split(
    news_df['article_paragraph'], news_df['type'],
    train_size = 0.80, random_state = 0
    )

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

# min_df, max_df and max_features help us get rid of less useful features
count_vector = CountVectorizer(stop_words = 'english',
                               min_df = 0.005,  # Ignore terms with frequancy higher than 0.005
                               max_df = 0.7,   # Ignore terms with frequancy lower than 0.7
                               max_features = 300,  # Keep top 300 ordered by term frequency
                               ngram_range=(1,2))


train_x = count_vector.fit_transform(X_train)
test_x = count_vector.transform(X_test)

train_x = train_x.toarray()
test_x = test_x.toarray()



class KNN:
  ### kNN classifier with choosen distance counter ###

  def __init__(self, k=1):
    self.k = k

  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    self.labels = np.unique(y_train)

  def predict(self, X_test):
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists)

  def compute_distances(self, x):
    """
    Inputs:
    - x: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the euclidean or hamming distance between the ith test point and the jth training
      point.
    """

    dists = []

#   Eucledian Distance

    for i in range(len(x)):
      dist = np.sqrt(np.sum((self.X_train - x[i])**2, axis=1))
      dists.append(dist)


#   Hamming Distance

    # for i in range(len(x)):
    #   dist = np.sum((self.X_train != x[i]), axis=1)
    #   dists.append(dist)

    return np.array(dists)


  def predict_labels(self, dists):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data
    """
    y_pred = []

    for i in range(len(dists)):
      indices = np.argsort(dists[i])[:self.k] # Select the K nearest neighbors from the distance matrix
                                              # Return the distances from points, sorted in ascending order

      labels, counts = np.unique(self.y_train[indices], return_counts = True) # Get the indexes of the nearest neighbors and count the number of matches
      idx = np.argmax(counts) # Returns the index of the maximum value
      pred = labels[idx]
      y_pred.append(pred)

    return np.array(y_pred)
  


'''
Testing with different params
'''

#Multinomial Naive Bayes Accuracy score:  1.0 (ngram_range=(1,2))

# naive_bayes = MultinomialNB()
# naive_bayes.fit(train_x, y_train)
# predictions = naive_bayes.predict(test_x)
# print("Accuracy score: ", accuracy_score(y_test, predictions))


#Gaussian Naive Bayes Accuracy score:  0.946 (ngram_range=(1,4))

# gaussian_bayes = GaussianNB()
# gaussian_bayes.fit(train_x, y_train)
# predictions = gaussian_bayes.predict(test_x)
# print("Accuracy score: ", accuracy_score(y_test, predictions))


#KNN Accuracy score:  0.906 (ngram_range=(1,2))
  
knn = KNN(5)
knn.fit(train_x, y_train)
predictions = knn.predict(test_x)
print("Accuracy score: ", accuracy_score(y_test, predictions))