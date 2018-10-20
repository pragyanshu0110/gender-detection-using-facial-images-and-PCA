# predicting gender using face images using PCA 
# accuracy on test set: 68.22033898305084 %
'''============================================================'''

import numpy as np
from functions import *
# sklearn
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

seed = 7
np.random.seed(seed)

k=100

# extracting data
data=test_data()
test_x_orig,test_y=data[0],data[1]

# normalization
test_x=normalization(test_x_orig)


# apply pca
pca=PCA(n_components=k)
pca.fit(test_x)
transformed_test_data=pca.transform(test_x)


# Load the classifier
cls = joblib.load("pragya.pkl")


#prediction
nbr = cls.predict(transformed_test_data)

#print(nbr)

#print(nbr.shape)
print('accuracy on test set:',np.mean(nbr==test_y)*100)

