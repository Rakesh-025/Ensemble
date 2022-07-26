
############################## problem3 #######################################
import pandas as pd

cocoa_data = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Ensemble\Coca_Rating_Ensemble.xlsx", encoding = 'utf-8' )
cocoa_data = cocoa_data.iloc[:,[0,2,3,4,5,6,7,8]]

#checking null 
null = pd.concat([cocoa_data.isnull().sum()], axis=1, keys=['Null Values' , " "])
null[null.sum(axis=1)>0]

#imputing null
cocoa_data['Bean_Type'] = cocoa_data['Bean_Type'].fillna(cocoa_data['Bean_Type'].mode()[0])
cocoa_data['Origin'] = cocoa_data['Origin'].fillna(cocoa_data['Origin'].mode()[0])

#converting type
cocoa_data['Review'] = cocoa_data['Review'].astype('str')
cocoa_data['Cocoa_Percent'] = cocoa_data['Cocoa_Percent'].str.rstrip('%').astype('float')/100
cocoa_data.info

#creating dummies
cocoa_data1 = pd.get_dummies(cocoa_data[['Review','Company_Location','Bean_Type','Origin']])
cocoa_data.drop(cocoa_data[['Review','Company_Location','Bean_Type','Origin']], axis=1, inplace=True)
cocoa_data = pd.concat([cocoa_data, cocoa_data1],axis=1)

# Input and Output Split
predictors = cocoa_data.loc[:, cocoa_data.columns!="Company"]
type(predictors)

target = cocoa_data["Company"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train_x, train_y)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test_y, cv_rf_clf_grid.predict(test_x))
accuracy_score(test_y, cv_rf_clf_grid.predict(test_x))

#Bagging technique
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(train_x, train_y)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(test_y, bag_clf.predict(test_x))
accuracy_score(test_y, bag_clf.predict(test_x))

# Evaluation on Training Data
confusion_matrix(train_y, bag_clf.predict(train_x))
accuracy_score(train_y, bag_clf.predict(train_x))

#Ada boosting technique
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(train_x, train_y)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(test_y, ada_clf.predict(test_x))
accuracy_score(test_y, ada_clf.predict(test_x))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

#Gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(train_x, train_y)

from sklearn.metrics import accuracy_score, confusion_matrix

#accuracy on test data
confusion_matrix(test_y, boost_clf.predict(test_x))
accuracy_score(test_y, boost_clf.predict(test_x))

#XG boosting
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

xgb_clf.fit(train_x, train_y)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(test_y, xgb_clf.predict(test_x))
accuracy_score(test_y, xgb_clf.predict(test_x))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(train_x, train_y)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(test_y, cv_xg_clf.predict(test_x))
grid_search.best_params_

#stacking classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

# Create the ensemble's base learners and meta learner
# Append base learners to a list
base_learners = []

# KNN classifier model
knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

# Decision Tree Classifier model
dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

# Multi Layered Perceptron classifier
mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)
base_learners.append(mlpc)

# Meta model using Logistic Regression
meta_learner = LogisticRegression(solver='lbfgs')

# Create the training meta data
# Create variables to store meta data and the targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(train_x[train_indices], train_y[train_indices])
        predictions = learner.predict_proba(train_x[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
    meta_index += len(test_indices)

# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(test_y, learner.predict(test_x))
    base_acc.append(acc)
    
test_meta_data = test_meta_data.transpose()

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')

#voting technique
#hard voting
# Import the required libraries
from sklearn import linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(train_x, train_y)

# Predict the most voted class
hard_predictions = voting.predict(test_x)

# Accuracy of hard voting
print('Hard Voting:', accuracy_score(test_y, hard_predictions))

# Soft Voting
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(train_x, train_y)

learner_4.fit(train_x, train_y)
learner_5.fit(train_x, train_y)
learner_6.fit(train_x, train_y)

# Predict the most probable class
soft_predictions = voting.predict(test_x)

# Get the base learner predictions
predictions_4 = learner_4.predict(test_x)
predictions_5 = learner_5.predict(test_x)
predictions_6 = learner_6.predict(test_x)

# Accuracies of base learners
print('L4:', accuracy_score(test_y, predictions_4))
print('L5:', accuracy_score(test_y, predictions_5))
print('L6:', accuracy_score(test_y, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(test_y, soft_predictions))