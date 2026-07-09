import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def train_baseline_model(X_train: pd.DataFrame, 
                         y_train: pd.Series, 
                         model_type: str = "logistic_regression",
                         random_state: int = 42) -> Pipeline:
    """
    Train a baseline model using the specified model type.

    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    model_type (str): The type of model to train. Options are 'logistic_regression' or 'random_forest'.
    random_state (int): The random state for reproducibility.

    Returns:
    Pipeline: The trained pipeline.
    """

    numeric_columns = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object', 'category', 'bool', 'str']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )   

    if model_type == "logistic_regression":
        model = LogisticRegression(random_state=random_state, max_iter=1000)

    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state)

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'logistic_regression' or 'random_forest'.")
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)

    return pipeline