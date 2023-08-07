from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle

# --------------------------------------------------------------
# Load X_train, X_test, y_train and Y_test
# --------------------------------------------------------------
X_train = pd.read_pickle("../../data/interim/X_train.pkl")
X_test = pd.read_pickle("../../data/interim/X_test.pkl")
y_train = pd.read_pickle("../../data/interim/y_train.pkl")
y_test = pd.read_pickle("../../data/interim/y_test.pkl")

# --------------------------------------------------------------
# Create the Model
# --------------------------------------------------------------

hidden_layer_sizes = (100,)
activation = "logistic"
max_iter = 2000
learning_rate = "adaptive"
alpha = 0.0001

nn = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    max_iter=max_iter,
    learning_rate=learning_rate,
    alpha=alpha,
)

# --------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------

nn.fit(
    X_train,
    y_train.values.ravel(),
)

# --------------------------------------------------------------
# Test the model
# --------------------------------------------------------------

pred_prob_training_y = nn.predict_proba(X_train)
pred_prob_test_y = nn.predict_proba(X_test)
pred_training_y = nn.predict(X_train)
pred_test_y = nn.predict(X_test)
frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

accuracy = accuracy_score(y_test, pred_test_y)

classes = frame_prob_test_y.columns

cm = confusion_matrix(y_test, pred_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.savefig("../../reports/figures/Confusion Matrix NN Model")
plt.show()

# --------------------------------------------------------------
# Dump the Model
# --------------------------------------------------------------

pickle.dump(nn, open("../../models/FeedForwardNeuralNetworkModel.pkl", "wb"))
