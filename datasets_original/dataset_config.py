# Dataset configuration for Adult, Bank, and Credit datasets

Adult = {
    "age": {"range": [1, 9], "NaN":[], "type":  "categorical", "sensitive": True},
    "workclass": {"range": [0, 7], "NaN":["100"], "type":  "categorical", "sensitive": False},
    "fnlwgt": {"range": [0, 74], "NaN":[], "type":  "numerical", "sensitive": False},
    "education": {"range": [0, 15], "NaN":[], "type":  "categorical", "sensitive": False},
    "marital_status": {"range": [0, 6], "NaN":[], "type":  "categorical", "sensitive": False},
    "occupation": {"range": [0, 13], "NaN":["100"], "type":  "categorical", "sensitive": False},
    "relationship": {"range": [0, 5], "NaN":[], "type":  "categorical", "sensitive": False},
    "race": {"range": [0, 4], "NaN":[], "type":  "categorical", "sensitive": True},
    "sex": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": True},
    "capital_gain": {"range": [0, 41], "NaN":[], "type":  "numerical", "sensitive": False},
    "capital_loss": {"range": [0, 43], "NaN":[], "type":  "numerical", "sensitive": False},
    "hours_per_week": {"range": [1, 99], "NaN":["99"], "type":  "numerical", "sensitive": False},
    "native_country": {"range": [0, 41], "NaN":[], "type":  "categorical", "sensitive": False},
    "Class": {"range": [0, 1], "NaN":[], "type":  "output", "sensitive": False},
}

Bank = {
    "age": {"range": [1, 9], "NaN":[], "type":  "categorical", "sensitive": True},
    "job": {"range": [0, 11], "NaN":[], "type":  "categorical", "sensitive": False},
    "marital": {"range": [0, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "education": {"range": [0, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "default": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    "balance": {"range": [-20, 179], "NaN":[], "type":  "numerical", "sensitive": False},
    "housing": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    "loan": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    "contact": {"range": [0, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "day": {"range": [1, 31], "NaN":[], "type":  "numerical", "sensitive": False},
    "month": {"range": [0, 11], "NaN":[], "type":  "categorical", "sensitive": False},
    "duration": {"range": [0, 99], "NaN":[], "type":  "numerical", "sensitive": False},
    "campaign": {"range": [1, 63], "NaN":[], "type":  "numerical", "sensitive": False},
    "pdays": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    "previous": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    "poutcome": {"range": [0, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "Class": {"range": [0, 1], "NaN":[], "type":  "output", "sensitive": False}
}

Credit = {
    "checking_status": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "duration": {"range": [4, 72], "NaN":[], "type":  "numerical", "sensitive": False},
    "credit_history": {"range": [0, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "purpose": {"range": [0, 10], "NaN":[], "type":  "categorical", "sensitive": False},
    "credit_amount": {"range": [2, 184], "NaN":[], "type":  "numerical", "sensitive": False},
    "savings_status": {"range": [1, 5], "NaN":[], "type":  "categorical", "sensitive": False},
    "employment": {"range": [1, 5], "NaN":[], "type":  "categorical", "sensitive": False},
    "installment_commitment": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "sex": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": True},
    "other_parties": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "residence": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "property_magnitude": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "age": {"range": [1, 7], "NaN":[], "type":  "numerical", "sensitive": True},
    "other_employment_plans": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "housing": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "existing_credits": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "job": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "own_telephone": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "telephone": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "foreign_worker": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "Class": {"range": [1, 2], "NaN":[], "type":  "output", "sensitive": False}
}