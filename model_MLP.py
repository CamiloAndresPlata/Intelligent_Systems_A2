# %% Imports
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef

# %% Load dataset and create train-test sets
data = load_wine()
X = data.data
X = minmax_scale(X)
y = data.target
var_names = data.feature_names
var_names = [var_names[i].title().replace('/','_') for i in range(0, len(var_names))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(14,14,14),random_state=42, max_iter=500)
regr.fit(X_train, y_train)

# %% Get model predictions
y_pred = regr.predict(X_test)

# %% Classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
precision = precision_score(y_test, y_pred, average="macro")
print("presicion: {:.3f}".format(precision))
f1 = f1_score(y_test, y_pred, average="macro")
print("f1 score: {:.3f}".format(f1))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))
matthews_coef = matthews_corrcoef(y_test, y_pred)
print("matthews correlation coefficient: {:.3f}".format(matthews_coef))
