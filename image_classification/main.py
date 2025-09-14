import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

import pickle

# prepare data
input_dir = "./clf-data"
categories = ['empty', 'not_empty']

date = []
labels = []

for category_inx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        date.append(img.flatten())
        labels.append(category_inx)

date = np.asarray(date)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(date, labels, test_size=0.2, stratify=labels, shuffle=True)

# train classifier
classifier = SVC()

parameters = [{'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}]

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters)
grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of samples are correctly classified'.format(str(score*100)))
print(classification_report(y_test, y_prediction))

# save classifier
pickle.dump(best_estimator, open('./models/model.p', 'wb'))