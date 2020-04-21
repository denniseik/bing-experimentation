import logging
import sys
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from helpers import plot_confusion_matrix
from azureml.core import Run

OUTPUTSFOLDER = "outputs"

# Get the Azure ML Run context
run = Run.get_context()

# Fetch data over HTTP
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extracting features from the training data using a sparse vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, 
                             max_df=0.5,
                             stop_words='english')

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)


def benchmark(clf, name=""):
    """benchmark classifier performance"""

    # create a child run for Azure ML logging
    child_run = run.child_run(name=name)
    child_run.log("ClassifierName", name)

    # train a classifier model provided as an argument
    print("\nTraining run with algorithm \n{}".format(clf))
    clf.fit(X_train, y_train)

    # evaluate on test set
    pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average='weighted')
    precision = metrics.precision_score(y_test, pred, average='weighted')
    recall = metrics.recall_score(y_test, pred, average='weighted')

    cm = metrics.confusion_matrix(y_test, pred, labels=clf.classes_)
    cm_plot = plot_confusion_matrix(cm, target_names=clf.classes_)

    # log evaluation metrics to AML
    child_run.log("accuracy", float(accuracy))
    child_run.log("f1", float(f1))
    child_run.log("precision", float(precision))
    child_run.log_image("Confusion Matrix {}".format(name), plot=cm_plot)

    # write model artifacts
    # create outputs folder if not exists
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)
        
    model_name = "model" + str(name) + ".pkl"
    filename = os.path.join(OUTPUTSFOLDER, model_name)
    joblib.dump(value=clf, filename=filename)

    # upload model artifact with child run
    child_run.upload_file(
        name=model_name,
        path_or_stream=filename
    )

    # mark the child run as completed
    child_run.complete()

#
# Run benchmark and collect results with multiple classifiers
#

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    
    # run benchmarking function for each
    benchmark(clf, name)


# Run with different regularization techniques
for penalty in ["l2", "l1"]:
    # Train Liblinear model
    name = penalty + "LinearSVC"
    benchmark(
        clf=LinearSVC(
            penalty=penalty,
            dual=False,
            tol=1e-3
        ),
        name=penalty + "LinearSVC"
    )

    # Train SGD model
    benchmark(
        SGDClassifier(
            alpha=.0001,
            max_iter=50,
            penalty=penalty
        ),
        name=penalty + "SGDClassifier"
    )

# Train SGD with Elastic Net penalty
benchmark(
    SGDClassifier(
        alpha=.0001,
        max_iter=50,
        penalty="elasticnet"
    ),
    name="Elastic-Net penalty"
)

# Train NearestCentroid without threshold
benchmark(
    NearestCentroid(),
    name="NearestCentroid (aka Rocchio classifier)"
)

# Train sparse Naive Bayes classifiers
benchmark(
    MultinomialNB(alpha=.01),
    name="Naive Bayes MultinomialNB"
)

benchmark(
    BernoulliNB(alpha=.01),
    name="Naive Bayes BernoulliNB"
)

benchmark(
    ComplementNB(alpha=.1),
    name="Naive Bayes ComplementNB"
)

# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
benchmark(
    Pipeline([
        ('feature_selection',
            SelectFromModel(
                LinearSVC(
                    penalty="l1",
                    dual=False,
                    tol=1e-3
                )
            )),
        ('classification',
            LinearSVC(penalty="l2"))
        ]
    ),
    name="LinearSVC with L1-based feature selection"
)