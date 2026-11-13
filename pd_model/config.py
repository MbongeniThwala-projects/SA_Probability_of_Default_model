'''
central rules for the propability of defualt (PD)

what counts as default, how ling we wathc a loan and where files live is kept here, 
so labels, tests and reports can never disagree with each other
'''
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
LOANS_PATH = RAW_DIR / "loan_performance.parquet"
PERFORMANCE_PATH = RAW_DIR / "loan_performance.parquet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LABELS_PATH = PROCESSED_DIR / "labels.parquet"
FEARURES_PATH = PROCESSED_DIR / "features.parquet"
MODEL_TABLE_PATH = PROCESSED_DIR / "model_table.parquet"

# Problem definition
DEFAULT_DPD_THRESHOLD = 90          # DAYS past due that counts as default (basel convent)
PERFORMANCE_WINDOW_MONTHS = 12      #how long after an obsevation point we watch for default
MIN_MONTHS_ON_BOOK = 3              # a loan needs a few months before we can score it
OBSERVATION_POINTS = [f"{year}-12-01" for year in range(2019, 2025)]
LOOKBACK_MONTHS = 6                 #for recent arrears window; the observation month + the 5 months before

TRAIN_POINTS = ["2019-12-01", "2020-12-01", "2021-12-01", "2022-12-01"]
VALIDATION_POINTS = ["2023-12-01"]
TEST_POINTS = ["2024-12-01"]


# synthetic data settings
RANDOM_SEED = 42
N_LOANS = 30_000
FIRST_ORIGINATION = "2019-01-01"
LAST_ORIGINATIOIN = "2023-12-01"
DATA_END_MONTH = "2025-12-01"

#ROUNDED approximations of provincial population shares
PROVINCES = {
    "Gauteng": 0.26,
    "KwaZulu-Natal": 0.19,
    "Western Cape": 0.12,
    "Eastern Cape": 0.11,
    "Limpopo": 0.10,
    "Mpumalanga": 0.08,
    "North West": 0.07,
    "Free State": 0.05,
    "Northern Cape": 0.02,
}

PRODUCTS = {
    "personal_loan": {
        "share": 0.50, 
        "terms": [24, 36, 48, 60],
        "income_multiple": (1, 8),      #loan amount as a multiple of MONTHLY income (low, high)
        "prime_margin": (2.0, 10.0),    #percentage points added to the prime landing rate (low, high)
        "monthly_prepay": .008          #chance per month that a performing borrower settles early
    },
    "vehicle_finance": {
        "share": .30,
        "terms": [48, 60, 72],
        "income_multiple": (6, 24),
        "prime_margin": (.0, 4.0),
        "monthly_prepay": .006
    },
    "home_loan": {
        "share": .20,
        "terms": [240, 300],
        "income_multiple": (30, 55),
        "prime_margin": (-1.0, 1.0),
        "monthly_prepay": .003
    }
}

# TODO: replace woth the monthly SARB prime landing rate and Stats SA
PRIME_RATE_BY_YEAR = {2019: 10.0, 2020: 8.0, 2021: 7.0, 2022: 8.5, 2023: 11.5, 2024: 11.75, 2025: 11.0}
UNEMPLOYMENT_BY_YEAR = {2019: 28, 2020: 30, 2021: 34, 2022: 33, 2023: 32, 2024: 33, 2025: 32}