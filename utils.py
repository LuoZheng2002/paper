from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, make_scorer
class SelectFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply_transform=False):
        self.apply_transform = apply_transform

    def fit(self, X, y=None):
        # Fit the transformer (if needed)
        return self

    def transform(self, X, y=None):
        if self.apply_transform:
            # Apply your transformation here
            return SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear')).fit_transform(X, y)
        return X  # No transformation applied
    
from sklearn.base import BaseEstimator, ClassifierMixin



class ModelSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def youdens_j(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # True positive rate
    specificity = tn / (tn + fp)  # True negative rate
    return sensitivity + specificity - 1

# Create custom scorers for unsupported cross-validation metrics
specificity_scorer = make_scorer(specificity)
youdens_j_scorer = make_scorer(youdens_j)