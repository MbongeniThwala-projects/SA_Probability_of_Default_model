'''
loans_static     -> one row per loan: who borrowed what, on which terms
loan_performance -> one row per loan per month: days past due, balance
'''

import numpy as np
import pandas as pd

from pd_model import config

# Hazard model settings
BASE_LOGIT = -5.2       # might have to tune this if the printed default is far outside the range noted in the summary
PRODUCT_RISK = {
    "personal_loan": .4, 
    "vehicle_finance": 0.0,
    "home_loan": -0.6
    }
EMPLOYMENT_RISK = {
    "permanent": 0.0,
    "contract": 0.5,
    "self_employed": .3
}

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_static_loans(rng):
    n = config.N_LOANS

    months = pd.date_range(config.FIRST_ORIGINATION, config.LAST_ORIGINATIOIN, freq="MS")
    origination_month = rng.choice(months.values, size=n)

    product_names = list(config.PRODUCTS)
    product = rng.choice(
        product_names,
        size=n,
        p=[config.PRODUCTS[p]["share"] for p in product_names]
    )
    province = rng.choice(list(config.PROVINCES), size=n, p=list(config.PROVINCES.values()))
    employment = rng.choice(list(EMPLOYMENT_RISK), size=n, p=[.65, 0.20, 0.15])

    age = np.clip(rng.normal(37, 9, n), 21, 65).round().astype(int)
    monthly_income = np.clip(
        np.exp(rng.normal(np.log(18_000), .55, n)), 4_000, 150_000
    ).round(-2)

    #Bureau score loosely tied to age and income, plus noice
    bureau_score = np.clip(
        600 + 2.0 * (age - 37) + 15 * np.log(monthly_income / 18_000) + rng.normal(0, 55, n), 300, 850
    ).round().astype(int)

    