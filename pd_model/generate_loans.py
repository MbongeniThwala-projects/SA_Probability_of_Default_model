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

    #amount, term and margin depend on the product, so fill them product by product
    term = np.zeros(n, dtype=int)
    loan_amount = np.zeros(n)
    margin = np.zeros(n)
    for name, spec in config.PRODUCTS.items():
        mask = product == name
        k = mask.sum()
        term[mask] = rng.choice(spec["terms"], size=k)
        low, high = spec["income_multiple"]
        loan_amount[mask] = monthly_income[mask] * rng.uniform(low, high, size=k)
        margin[mask] = rng.uniform(*spec["prime_margin"], size=k)

    #rate = prime at origination + product margin + markup for weaker bureau scores
    # simplification: rate stays fixed for the life of the loan
    origination_year = pd.DatetimeIndex(origination_month).year
    prime = np.array([config.PRIME_RATE_BY_YEAR[y] for y in origination_year])
    risk_markup = np.clip((600 - bureau_score) / 100, 0, 3)
    annual_rate = prime + margin + risk_markup

    monthly_rate = annual_rate / 100 / 12
    instalment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term)

    # TODO: "write" the dataframe frame
    return pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(1, n + 1)],
    })

def _static_risk(loans):
    '''The part of default risk that depends only on who the borrows is.'''
    return (
        -0.012 * (loans["bureau_score"].to_numpy() - 620)
        + 3.0 * (loans["payment_to_income"].to_numpy() - 0.20)
        - 0.02 * (loans["age"].to_numpy() -37)
        + loans["employment_status"].map(EMPLOYMENT_RISK).to_numpy()
        + loans["products"].map(PRODUCT_RISK).to_numpy()
    )
    
def simulate_performance(loans, rng):
    n = len(loans)
    loan_ids = loans["loan_id"].to_numpy()
    term = loans["term_months"].to_numpy()
    amount = loans["loan_amount"].to_numpy()
    monthly_rate = loans["interest_rate"].to_numpy() / 100 / 12
    prepay_prob = loans["product"].map(
        {name: spec["monthly_prepay"] for name, spec in config.PRODUCTS.items()}
    ).to_numpy()
    static_risk = _static_risk(loans)

    origination = pd.DatetimeIndex(loans["origination_month"])
    origination_idx = (origination.year * 12 + origination.month).to_numpy()

    dpd = np.zeros(n, dtype=int)        #current days past due per loan
    closed = np.zeros(n, dtype=bool)    #defaulted, matured or prepaid
    frames = []

    for month in pd.date_range(config.FIRST_ORIGINATION, config.DATA_END_MONTH, freq="MS"):
        mob = (month.year * 12 + month.month) - origination_idx         #months on book
        active = np.flatnonzero((~closed) & (mob >= 1) & (mob <= term))
        if active.size == 0:
            continue

        #risk this month = who the borrower is + the economy + loan age
        risk = (
            static_risk[active]
            + 0.15 * (config.UNEMPLOYMENT_BY_YEAR[month.year] - 30)
            + .12 * (config.PRIME_RATE_BY_YEAR[month.year] - 9)
            + .6 * np.exp(-(((mob[active] - 14) / 10) ** 2))            #risk peaks around month 14
        )
        p_miss = _sigmoid(BASE_LOGIT + risk)            #performing -> 30 days late
        p_roll_30 = sigmoid(-1.5 + 0.5 * risk)          #30 to 60 days late
        p_roll_60 = _sigmoid(-.5 + .5 * risk)           # 60 -> 90 days late (default)

        draw = rng.random(active.size)
        previous = dpd[active]
        new = np.zeros(active.size, dtype=int)          #anything that does not roll worse cures to 0
        new[(previous == 0) & (draw < p_miss)] = 30
        new[(previous == 30) & (draw < p_roll_30)] = 60
        new[(previous == 60) & (draw < p_roll_60)] = 90
        dpd[active] = new

        #Outstanding balance from the standard amortisation formula
        r = monthly_rate[active]
        growth = (1 + r) ** term[active]
        balance = amount[active] * (growth - (1 + r) ** mob[active]) / (growth -1)

        frames.append(pd.DataFrame({
            "loan_id": loan_ids[active],
            "snapshot_month": month,
            "months_on_book": mob[active],
            "days_past_due": new,
            "outstanding_balance": balance.round(2)
        }))

        defaulted = new >= config.DEFAULT_DPD_THRESHOLD
        matured = mob[active] >= term[active]
        prepaid = (new == 0) & (rng.random(active.size) < prepay_prob[active])
        closed[active[defaulted | matured | prepaid]] = True

    return pd.concat(frames, ignore_index=True)

def _print_summary(loans, performance):
    ever_default = performance.groupby("loan_id")["days_past_due"].max() >= config.DEFAULT_DPD_THRESHOLD
    print(f'Loans: {len(loans):,}')
    print(f"Performance rows: {len(performance):,}")
    print(f"Share of loans that ever reach {config.DEFAULT_DPD_THRESHOLD}+ DPD: {ever_default.mean():.1%}")
    print(performance["days_past_due"].value_counts(normalize=True).sort_index().round(4))

def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    loans = generate_static_loans(rng)
    performance = simulate_performance(loans, rng)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(config.LOANS_PATH, index=False)
    performance.to_parquet(config.PERFORMANCE_PATH, index=False)
    _print_summary(loans, performance)

    # TODO: edge cases, loans that default in their first months, loans closed before the window ends

if __name__ == "__main__":
    main()