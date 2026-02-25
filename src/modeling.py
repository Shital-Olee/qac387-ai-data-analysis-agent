from typing import Optional, List, Dict, Any
import pandas as pd

def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    (Student task): Fit a multiple linear regression model.

    Requirements:
    - Outcome must be numeric; raise ValueError otherwise
    - If predictors is None:
        use ALL numeric columns except outcome
    - Drop rows with missing values in outcome or predictors before fitting
    - Fit the model using least squares:
        y = intercept + b1*x1 + b2*x2 + ...
    - Return a JSON-safe dictionary containing:
        outcome, predictors, n_rows_used, r_squared, adj_r_squared,
        intercept, coefficients (dict)

    Hints: use statsmodels package:
    import statsmodels.api as sm
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()

    IMPORTANT:
    - Convert any numpy/pandas scalars to Python floats/ints before returning.
    """
    import statsmodels.api as sm
    
    if outcome not in df.columns: 
        raise ValueError(f"Outcome column '{outcome}' not found in the dataframe.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError("Outcome variable must be numeric.")
    if predictors is None: 
        numeric_cols = df.select_dtypes(include = ["number"]).columns.tolist()
        predictors = [c for c in numeric_cols if c != outcome]
    if not predictors: 
        raise ValueError("No valid predictors were provided.")
    
    model_df = df[[outcome] + predictors].dropna()

    if model_df.empty: 
        raise ValueError("No rows remaining after dropping missing values.")
    
    X = model_df[predictors]
    X = sm.add_constant(X)
    y = model_df[outcome]

    model = sm.OLS(y, X).fit()

    coef_dict = {
        str(k): float(v)
        for k, v in model.params.items()
        if k != "const"
    }

    results = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": float(model.params["const"]),
        "coefficients": coef_dict
    }
    return results