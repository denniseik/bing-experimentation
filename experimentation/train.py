import sys
import numpy as np
import argparse
from optparse import OptionParser
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from azureml.core import Run
from helpers import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    '--max-depth',
    type=int,
    dest='max_depth',
    default=None,
    help='random forest max depth'
)
parser.add_argument(
    '--min-samples-leaf',
    type=int,
    dest='min_sample_leaf',
    default=1,
    help='random forest minimum samples per leaf'
)
args = parser.parse_args()

# Get the Azure ML Run context
run = Run.get_context()

# Fetch data over HTTP
print("Loading 20 newsgroups dataset..")

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

data_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)

data_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42
)

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

# Extract features from the training data using a sparse vectorizer
vectorizer = TfidfVectorizer(
    sublinear_tf=True, 
    max_df=0.5,
    stop_words='english'
)

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

def benchmark(clf, name=""):
    """benchmark classifier performance"""

    # train a model
    print("\nTraining run with algorithm {} \n{}".format(name, clf))
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
    run.log("accuracy", float(accuracy))
    run.log("f1", float(f1))
    run.log("precision", float(precision))
    run.log_image("Confusion Matrix {}".format(name), plot=cm_plot)

    # write model artifact to AML
    model_name = "model" + str(name) + ".pkl"
    filename = "outputs/" + model_name
    joblib.dump(value=clf, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)


# Run benchmark and collect results with multiple classifiers
benchmark(
    clf=RandomForestClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_sample_leaf
    ),
    name="RandomForestClassifier"
)

# Mark the AML run as complete
run.complete()