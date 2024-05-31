import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from category_encoders import CatBoostEncoder

class MissingValuesFeatureRemover(BaseEstimator, TransformerMixin):
    """
    Removes features with missing values above a fractional threshold, defaulting to 0.2 
    """
    
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.features_to_drop = []
        self.output = "pandas"
        self.fitted = False
    
    def fit(self, X, y=None):
        nan_fracs = X.isna().sum() / X.shape[0]
        
        self.features_to_drop = nan_fracs[nan_fracs >= self.threshold].keys().to_list()
        
        self.fitted = True
        
        return self
        
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Fit the transformer first using fit().")
            
        cleaned_X = X.drop(self.features_to_drop, axis=1)
            
        return cleaned_X if self.output == "pandas" else cleaned_X.to_numpy()
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def set_output(self, transform="pandas"):
        self.output = transform

class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features, by performing dummy encoding if they have 3 or less categories,
    otherwise catboost encoding.
    """
    
    def __init__(self):
        self.binary_encoder = OneHotEncoder(drop="first")
        self.nominal_encoder = CatBoostEncoder()
        self.output = "pandas"
        self.fitted = False

    def set_features_and_categories(self, X):
        self.binary_features, self.nominal_features = [],[]
        
        for col in X:
                
            if X[col].nunique() <= 3:
                self.binary_features.append(col)
            else:
                self.nominal_features.append(col)
    
    def fit(self, X, y=None):
        
        self.set_features_and_categories(X)
        
        if self.binary_features:
            self.binary_encoder.fit(X[self.binary_features])
        if self.nominal_features:
            self.nominal_encoder.fit(X[self.nominal_features], y)
            
        self.fitted = True
        
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Fit the transformer first using fit()")
        
        result = X.drop(columns=self.binary_features + self.nominal_features, inplace=False)
        
        if self.binary_features:
            binary_encoded = self.binary_encoder.transform(X[self.binary_features]).toarray()
            binary_cols = self.binary_encoder.get_feature_names_out(self.binary_features)
            result = pd.concat([result, pd.DataFrame(binary_encoded, columns=binary_cols, index=X.index)], axis=1)
        
        if self.nominal_features:
            nominal_encoded = self.nominal_encoder.transform(X[self.nominal_features])
            result = pd.concat([result, nominal_encoded], axis=1)
        
        return result if self.output == "pandas" else result.to_numpy()
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)
    
    def set_output(self, transform="pandas"):
        self.output = transform