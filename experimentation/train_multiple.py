import os
import numpy as np
import argparse
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn.externals import joblib

OUTPUTSFOLDER = "outputs/"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--alphas',
    type=int,
    dest='alphas',
    default=np.arange(0.0, 1.0, 0.05),
    help='alphas'
)
args = parser.parse_args()

# Get the Azure ML Run context
run = Run.get_context()

# Load the diabetes sample data set
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

for alpha in args.alphas:
    # Use Ridge algorithm to create a regression model
    child_run = run.child_run()

    reg = Ridge(alpha=alpha)
    reg.fit(data["train"]["X"], data["train"]["y"])

    # evaluate on the test set
    predictions = reg.predict(data["test"]["X"])
    mse = mean_squared_error(predictions, data["test"]["y"])
    r2 = r2_score(predictions, data["test"]["y"])
    
    # log evaluation metrics to Azure ML
    child_run.log('alpha', float(alpha))
    child_run.log('mse', float(mse))
    child_run.log('r2', float(r2))

    # save model in the outputs folder so it automatically get uploaded
    os.makedirs(OUTPUTSFOLDER, exist_ok=True)
    model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    model_path = os.path.join(OUTPUTSFOLDER, model_file_name)

    with open(model_file_name, "wb") as file:
        joblib.dump(value=reg, filename=model_path)
        child_run.upload_file("Classifier", model_path)

    print('alpha is {0:.2f}, and mse is {1:0.2f}'.format(alpha, mse))

    child_run.complete()

# mark the AML run as complete
run.complete()