# modified by referring to RSUTT' config file
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
    # Age binned into numerical categories (e.g., 1=18-25)
    "age": {"range": [1, 9], "NaN":[], "type":  "categorical", "sensitive": True},
    # 12 job categories (0-11)
    "job": {"range": [0, 11], "NaN":[], "type":  "categorical", "sensitive": False},
    # Marital status: 0=single, 1=married, 2=divorced, etc.
    "marital": {"range": [0, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    # Education: 4 categories (e.g., primary, secondary, tertiary, unknown)
    "education": {"range": [0, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    # Default on loan (0=no, 1=yes)
    "default": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    # Bank balance (already binned or normalized)
    "balance": {"range": [-20, 179], "NaN":[], "type":  "numerical", "sensitive": False},
    # Housing loan (0=no, 1=yes)
    "housing": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    # Personal loan (0=no, 1=yes)
    "loan": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    # Contact method (e.g., cellular, telephone, unknown)
    "contact": {"range": [0, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    # Last contact day of the month (1-31)
    "day": {"range": [1, 31], "NaN":[], "type":  "numerical", "sensitive": False},
    # Month (0=Jan, ..., 11=Dec)
    "month": {"range": [0, 11], "NaN":[], "type":  "categorical", "sensitive": False},
    # Duration of the last contact in seconds (99 may represent missing/compressed)
    "duration": {"range": [0, 99], "NaN":[], "type":  "numerical", "sensitive": False},
    # Number of contacts performed during this campaign
    "campaign": {"range": [1, 63], "NaN":[], "type":  "numerical", "sensitive": False},
    # Number of days since the client was last contacted in previous campaigns
    "pdays": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    # Number of contacts performed before this campaign
    "previous": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": False},
    # Outcome of the previous marketing campaign (0-3)
    "poutcome": {"range": [0, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    # Final response (0=No, 1=Yes)
    "Class": {"range": [0, 1], "NaN":[], "type":  "output", "sensitive": False}
}

Credit = {
    "checking_status": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "duration": {"range": [4, 72], "NaN":[], "type":  "numerical", "sensitive": False},
    "credit_history": {"range": [0, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "purpose": {"range": [0, 10], "NaN":[], "type":  "categorical", "sensitive": False},
    "credit_amount": {"range": [2, 184], "NaN":[], "type":  "numerical", "sensitive": False},
    "savings_status": {"range": [1, 5], "NaN":[], "type":  "categorical", "sensitive": False},
    "emplyoment": {"range": [1, 5], "NaN":[], "type":  "categorical", "sensitive": False},
    "installment_commitment": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "sex": {"range": [0, 1], "NaN":[], "type":  "categorical", "sensitive": True},
    # "personal_status": {"range": [0, 4], "NaN":[], "type":  categorical, "sensitive": False}
    "other_parties": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "residence": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "property_magnitude": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "age": {"range": [1, 7], "NaN":[], "type":  "numerical", "sensitive": True},
    "other_emplyoment_plans": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "housing": {"range": [1, 3], "NaN":[], "type":  "categorical", "sensitive": False},
    "existing_credits": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "job": {"range": [1, 4], "NaN":[], "type":  "categorical", "sensitive": False},
    "own_telephone": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "telephone": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "foreign_worker": {"range": [1, 2], "NaN":[], "type":  "categorical", "sensitive": False},
    "Class": {"range": [1, 2], "NaN":[], "type":  "output", "sensitive": False}
}


# adult_dataset = [
#     {"name": "age", "range": [0, 9], "NaN": [], "type": "numerical", "sensitive": True},
#     {"name": "workclass", "range": [0, 7], "NaN": ["100"], "type": "categorical", "sensitive": False},
#     {"name": "fnlwgt", "range": [0, 74], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "education", "range": [0, 15], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "marital_status", "range": [0, 6], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "occupation", "range": [0, 13], "NaN": ["100"], "type": "categorical", "sensitive": False},
#     {"name": "relationship", "range": [0, 5], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "race", "range": [0, 4], "NaN": [], "type": "categorical", "sensitive": True},
#     {"name": "gender", "range": [0, 1], "NaN": [], "type": "categorical", "sensitive": True},
#     {"name": "capital_gain", "range": [0, 41], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "capital_loss", "range": [0, 43], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "hours_per_week", "range": [1, 99], "NaN": ["99"], "type": "numerical", "sensitive": False},
#     {"name": "native_country", "range": [0, 41], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "Class", "range": [0, 1], "NaN": [], "type": "output", "sensitive": False},
# ]
#
# bank_dataset = [
#     # Age binned into numerical categories (e.g., 1=18-25)
#     {"name": "age", "range": [1, 9], "NaN": [], "type": "numerical", "sensitive": True},
#     # 12 job categories (0-11)
#     {"name": "job", "range": [0, 11], "NaN": [], "type": "numerical", "sensitive": False},
#     # Marital status: 0=single, 1=married, 2=divorced, etc.
#     {"name": "marital", "range": [0, 2], "NaN": [], "type": "numerical", "sensitive": False},
#     # Education: 4 categories (e.g., primary, secondary, tertiary, unknown)
#     {"name": "education", "range": [0, 3], "NaN": [], "type": "numerical", "sensitive": False},
#     # Default on loan (0=no, 1=yes)
#     {"name": "default", "range": [0, 1], "NaN": [], "type": "numerical", "sensitive": False},
#     # Bank balance (already binned or normalized)
#     {"name": "balance", "range": [-20, 179], "NaN": [], "type": "numerical", "sensitive": False},
#     # Housing loan (0=no, 1=yes)
#     {"name": "housing", "range": [0, 1], "NaN": [], "type": "numerical", "sensitive": False},
#     # Personal loan (0=no, 1=yes)
#     {"name": "loan", "range": [0, 1], "NaN": [], "type": "numerical", "sensitive": False},
#     # Contact method (e.g., cellular, telephone, unknown)
#     {"name": "contact", "range": [0, 2], "NaN": [], "type": "numerical", "sensitive": False},
#     # Last contact day of the month (1-31)
#     {"name": "day", "range": [1, 31], "NaN": [], "type": "numerical", "sensitive": False},
#     # Month (0=Jan, ..., 11=Dec)
#     {"name": "month", "range": [0, 11], "NaN": [], "type": "numerical", "sensitive": False},
#     # Duration of the last contact in seconds (99 may represent missing/compressed)
#     {"name": "duration", "range": [0, 99], "NaN": [], "type": "numerical", "sensitive": False},
#     # Number of contacts performed during this campaign
#     {"name": "campaign", "range": [1, 63], "NaN": [], "type": "numerical", "sensitive": False},
#     # Number of days since the client was last contacted in previous campaigns
#     {"name": "pdays", "range": [0, 1], "NaN": [], "type": "numerical", "sensitive": False},
#     # Number of contacts performed before this campaign
#     {"name": "previous", "range": [0, 1], "NaN": [], "type": "numerical", "sensitive": False},
#     # Outcome of the previous marketing campaign (0-3)
#     {"name": "poutcome", "range": [0, 3], "NaN": [], "type": "numerical", "sensitive": False},
#     # Final response (0=No, 1=Yes)
#     {"name": "Class", "range": [0, 1], "NaN": [], "type": "output", "sensitive": False}
# ]
#
#
# credit_dataset = [
#     {"name": "account_status", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "duration_in_month", "range": [4, 72], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "credit_history", "range": [1, 5], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "purpose", "range": [1, 10], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "credit_amount", "range": [250, 18424], "NaN": [], "type": "numerical", "sensitive": False},
#     {"name": "savings_status", "range": [1, 5], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "employment_since", "range": [1, 5], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "installment_commitment", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "gender", "range": [0, 1], "NaN": [], "type": "categorical", "sensitive": True},
#     # {"name": "personal_status", "range": [0, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "other_parties", "range": [1, 3], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "residence_since", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "property", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "age", "range": [19, 75], "NaN": [], "type": "numerical", "sensitive": True},
#     {"name": "other_installment_plans", "range": [1, 3], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "housing", "range": [1, 3], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "num_credits", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "job", "range": [1, 4], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "num_dependent", "range": [1, 2], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "telephone", "range": [1, 2], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "foreign_worker", "range": [1, 2], "NaN": [], "type": "categorical", "sensitive": False},
#     {"name": "Class", "range": [0, 1], "NaN": [], "type": "output", "sensitive": False}
# ]

