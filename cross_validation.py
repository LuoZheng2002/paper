import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy.typing as npt
from sklearn import clone, metrics
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, auc, precision_score, f1_score, roc_auc_score, 
    average_precision_score, recall_score, confusion_matrix, roc_curve
)


def calculate_metric(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric: str
) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred, average='binary', zero_division=0)
    elif metric == "f1-score":
        return f1_score(y_true, y_pred, average='binary')
    elif metric == "auroc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        return average_precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan
    elif metric == "specificity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        raise ValueError(f"Unknown metric: {metric}")
    


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    metric: str = "accuracy",
    bootstrap: bool=True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X, using 1,000 
    bootstrapped samples of the test set if bootstrap is set to True. Otherwise,
    returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.
    
    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision', 
                'sensitivity', and 'specificity')
    Returns:
        if bootstrap is True: the median performance and the empirical 95% confidence interval in np.float64
        if bootstrap is False: peformance 
    """
    # TODO: Implement this function
    # This is an optional but VERY useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    
    
    # Check if metric requires probabilities (e.g., auroc, average_precision)
    if metric in ["auroc", "average_precision"] and isinstance(clf_trained, LogisticRegression):
        # If predict_proba is not available, assume it's a regressor with continuous output
        y_pred = clf_trained.decision_function(X)
    else:
        y_pred = clf_trained.predict(X)
        y_pred = np.where(y_pred >= 0, 1, -1)

    # Single evaluation without bootstrapping
    if not bootstrap:
        return calculate_metric(y_true, y_pred, metric)
    
    # Bootstrap sampling to estimate performance and confidence interval
    n_bootstraps = 1000
    bootstrapped_scores = []

    # Perform bootstrapping
    for _ in range(n_bootstraps):
        # Resample data
        # to do: replace this with np.random.choice
        X_resampled, y_resampled = resample(X, y_true)
        
        # Predict on resampled data
        if metric in ["auroc", "average_precision"] and isinstance(clf_trained, LogisticRegression):
            y_pred_resampled = clf_trained.decision_function(X_resampled)
        else:
            y_pred_resampled = clf_trained.predict(X_resampled)
            y_pred_resampled = np.where(y_pred_resampled >= 0, 1, -1)
        # Calculate metric for resampled data
        score = calculate_metric(y_resampled, y_pred_resampled, metric)
        bootstrapped_scores.append(score)
    
    # Convert to numpy array for easier manipulation
    bootstrapped_scores = np.array(bootstrapped_scores)
    
    # Calculate median and 95% CI (2.5th and 97.5th percentiles)
    lower_bound = np.percentile(bootstrapped_scores, 2.5)
    upper_bound = np.percentile(bootstrapped_scores, 97.5)
    median_score = np.median(bootstrapped_scores)

    return median_score, lower_bound, upper_bound

def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # TODO: Implement this function

    # NOTE: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # TODO: Return the average, min,and max performance scores across all fold splits in a size 3 tuple.

    skf = StratifiedKFold(n_splits=k)
    performance_scores = []

    # Iterate over each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # Split data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Clone the classifier to ensure independence between folds
        clf_clone = clone(clf)
        
        # Train the classifier on the training data
        clf_clone.fit(X_train, y_train)
        
        # Evaluate performance on the test data
        score = performance(
            clf_trained=clf_clone,
            X=X_test,
            y_true=y_test,
            metric=metric,
            bootstrap=False  # Set bootstrap to False to get a single score
        )
        
        # Append the score to the list
        performance_scores.append(score)
        
        # Debugging statements (can be removed in production)
        # print(f"Fold {fold}: {metric} = {score}")
    
    # Calculate mean, min, and max of the performance scores
    mean_score = float(np.mean(performance_scores))
    min_score = float(np.min(performance_scores))
    max_score = float(np.max(performance_scores))
    
    # Debugging statements (can be removed in production)
    # print(f"Cross-Validation {metric} - Mean: {mean_score}, Min: {min_score}, Max: {max_score}")
    performance_tuple = (mean_score, min_score, max_score)
    return performance_tuple