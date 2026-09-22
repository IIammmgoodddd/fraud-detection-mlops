"""Feature engineering for the IEEE-CIS fraud detection dataset.

Input: merged transaction+identity frame; output: an all-float32 feature matrix.
Usage: `train, val = split_by_time(df)`; `fe = FeatureEngineer().fit(train)`;
`fe.transform(train)` / `fe.transform(val)`. Everything learned from the data is
fitted on train only, so validation can never influence a transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "isFraud"

SECONDS_PER_HOUR = 60 * 60
HOURS_PER_DAY = 24
SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR
DAYS_PER_WEEK = 7

TRAIN_END_DAY = 122
VALIDATION_GAP_DAYS = 30

AMOUNT_DECIMALS = 4
AMOUNT_SCALE = 10**AMOUNT_DECIMALS

CORRELATION_THRESHOLD = 0.9
CORRELATION_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

EMAIL_BOTH_MISSING = 0
EMAIL_RECIPIENT_MISSING = 1
EMAIL_PURCHASER_MISSING = 2
EMAIL_SAME_DOMAIN = 3
EMAIL_DIFFERENT_DOMAIN = 4

# fmt: off
ID_COLUMNS = ["TransactionID", "TransactionDT", "uid"]
LOW_CARDINALITY = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_34", "id_35",
    "id_36", "id_37", "id_38",
]
# Numeric ones stay in the matrix beside their frequency encoding: card1 and
# addr1 are codes whose ordering is meaningful (issuer ranges), so a count is
# not a replacement for the value.
HIGH_CARDINALITY_NUMERIC = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
HIGH_CARDINALITY_TEXT = [
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33",
]
HIGH_CARDINALITY = HIGH_CARDINALITY_NUMERIC + HIGH_CARDINALITY_TEXT
EMAIL_SUFFIX_COLUMNS = ["p_email_suffix", "r_email_suffix"]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]

# D9 holds a fraction of a day rather than a day count, so `day - D9` would
# just restate the time index.
D_OFFSET_COLUMNS = [column for column in D_COLUMNS if column != "D9"]

# A raw time index cannot generalise: scoring always happens after the last
# day the model was trained on, so every split on it sends new rows one way.
TIME_INDEX_COLUMNS = ["day", "TransactionDay"]

REQUIRED_COLUMNS = [
    "TransactionDT", "TransactionAmt", "P_emaildomain", "R_emaildomain",
    "card1", "addr1", "D1", "DeviceInfo",
]
# fmt: on


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Input frame is missing required columns: {missing}")


def split_by_time(
    df: pd.DataFrame,
    train_end_day: int = TRAIN_END_DAY,
    gap_days: int = VALIDATION_GAP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest days; validate after a gap, as the Kaggle test set does."""
    day = df["TransactionDT"] // SECONDS_PER_DAY
    train = df.loc[day <= train_end_day].copy()
    val = df.loc[day > train_end_day + gap_days].copy()
    return train, val


# --- Stateless (row-local) features ---


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    seconds = df["TransactionDT"]
    df["day"] = seconds // SECONDS_PER_DAY
    df["hour"] = (seconds // SECONDS_PER_HOUR) % HOURS_PER_DAY
    df["dayofweek"] = df["day"] % DAYS_PER_WEEK
    return df


def count_decimal_places(fractional_part: pd.Series) -> np.ndarray:
    """Decimals ignoring trailing zeros: 0.5 -> 1, 0.125 -> 3, 0.0 and NaN -> 0."""
    digits = np.rint(
        np.nan_to_num(fractional_part.to_numpy(dtype="float64")) * AMOUNT_SCALE
    ).astype(np.int64)
    # Each trailing zero in the scaled integer is one decimal place fewer.
    has_nonzero_tail = [
        digits % 10**power != 0 for power in range(1, AMOUNT_DECIMALS + 1)
    ]
    place_counts = list(range(AMOUNT_DECIMALS, 0, -1))
    return np.select(has_nonzero_tail, place_counts, default=0).astype("int8")


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    amount = df["TransactionAmt"]
    fractional_part = (amount - np.floor(amount)).round(AMOUNT_DECIMALS)
    df["amt_log"] = np.log1p(amount)
    df["amt_is_round"] = (fractional_part == 0).astype("int8")
    # Three decimal places usually means the amount was converted from another currency.
    df["amt_decimals"] = count_decimal_places(fractional_part)
    return df


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    purchaser, recipient = df["P_emaildomain"], df["R_emaildomain"]
    df["email_match_state"] = np.select(
        [
            purchaser.isna() & recipient.isna(),
            purchaser.notna() & recipient.isna(),
            purchaser.isna() & recipient.notna(),
            purchaser == recipient,
        ],
        [
            EMAIL_BOTH_MISSING,
            EMAIL_RECIPIENT_MISSING,
            EMAIL_PURCHASER_MISSING,
            EMAIL_SAME_DOMAIN,
        ],
        default=EMAIL_DIFFERENT_DOMAIN,
    ).astype("int8")
    df["p_email_suffix"] = purchaser.str.rsplit(".", n=1).str[-1]
    df["r_email_suffix"] = recipient.str.rsplit(".", n=1).str[-1]
    return df


def add_entity_id(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate a cardholder; D1 is days since first use, so day - D1 pins its start."""
    df["card_start_day"] = df["day"] - df["D1"]
    df["uid"] = (
        df["card1"].astype("string")
        + "_"
        + df["addr1"].astype("string")
        + "_"
        + df["card_start_day"].astype("string")
    )
    return df


def add_normalized_d_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn D* offsets into the absolute day the event they count happened."""
    for column in D_OFFSET_COLUMNS:
        if column in df.columns:
            df[f"{column}_norm"] = df["day"] - df[column]
    return df


def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature that depends on one row only, so it is safe on any split."""
    validate_required_columns(df)
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_email_features(df)
    df = add_entity_id(df)
    return add_normalized_d_columns(df)


# --- Column selection ---


def find_v_columns(df: pd.DataFrame) -> list[str]:
    """The anonymised Vesta columns, V1 through V339."""
    names = df.columns.astype(str)
    return [name for name in names if name.startswith("V") and name[1:].isdigit()]


def find_missingness_blocks(df: pd.DataFrame, columns: list[str]) -> list[list[str]]:
    """Group columns by identical null count; V* columns arrive in a few such blocks."""
    blocks: dict[int, list[str]] = {}
    for column, null_count in df[columns].isna().sum().items():
        blocks.setdefault(int(null_count), []).append(str(column))
    return [blocks[null_count] for null_count in sorted(blocks)]


def select_decorrelated(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = CORRELATION_THRESHOLD,
    sample_size: int = CORRELATION_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """Greedily keep columns that aren't near-duplicates of an earlier one."""
    complete_rows = df[columns].dropna()
    if complete_rows.empty:
        return list(columns)
    if len(complete_rows) > sample_size:
        complete_rows = complete_rows.sample(sample_size, random_state=random_state)

    correlation = complete_rows.corr().abs().to_numpy()
    kept_positions: list[int] = []
    for position in range(len(columns)):
        # A NaN correlation (constant column) compares False, i.e. "not redundant".
        if not np.any(correlation[position, kept_positions] > threshold):
            kept_positions.append(position)
    return [columns[position] for position in kept_positions]


# --- Fitted pipeline ---


@dataclass
class FeatureEngineer:
    """Fit on train, transform anything. Never fit on validation or test."""

    corr_threshold: float = CORRELATION_THRESHOLD
    random_state: int = RANDOM_STATE

    v_blocks_: list[list[str]] = field(default_factory=list, init=False)
    v_kept_: list[str] = field(default_factory=list, init=False)
    frequency_maps_: dict[str, pd.Series] = field(default_factory=dict, init=False)
    category_maps_: dict[str, dict[object, int]] = field(
        default_factory=dict, init=False
    )
    entity_stats_: pd.DataFrame | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, train: pd.DataFrame) -> FeatureEngineer:
        prepared = add_stateless_features(train.copy())

        self.v_blocks_ = find_missingness_blocks(prepared, find_v_columns(prepared))
        self.v_kept_ = [
            column
            for block in self.v_blocks_
            for column in select_decorrelated(
                prepared, block, self.corr_threshold, random_state=self.random_state
            )
        ]
        self._fit_encodings(prepared)
        self.entity_stats_ = prepared.groupby("uid")["TransactionAmt"].agg(
            uid_count="size", uid_amt_mean="mean", uid_amt_std="std"
        )
        self.feature_names_ = self._build(prepared).columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("Call fit() on the training split before transform().")
        features = self._build(add_stateless_features(df.copy()))
        return features.reindex(columns=self.feature_names_)

    def _fit_encodings(self, prepared: pd.DataFrame) -> None:
        self.frequency_maps_.clear()
        self.category_maps_.clear()
        for column in HIGH_CARDINALITY:
            if column in prepared.columns:
                self.frequency_maps_[column] = prepared[column].value_counts()
        # Email suffixes need a fitted map like any other category: deriving codes
        # per call renumbers them whenever a split lacks or adds a suffix.
        for column in LOW_CARDINALITY + EMAIL_SUFFIX_COLUMNS:
            if column in prepared.columns:
                categories = prepared[column].dropna().unique()
                self.category_maps_[column] = {
                    value: code
                    for code, value in enumerate(sorted(categories, key=str))
                }

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        df["has_identity"] = df["DeviceInfo"].notna().astype("int8")
        for index, block in enumerate(self.v_blocks_):
            df[f"v_block{index}_missing"] = df[block[0]].isna().astype("int8")

        if self.entity_stats_ is not None:
            aligned_stats = self.entity_stats_.reindex(df["uid"])
            for stat_name in self.entity_stats_.columns:
                df[stat_name] = aligned_stats[stat_name].to_numpy()
            # A uid averaging to zero (a purchase and its refund) would divide to infinity.
            uid_mean = df["uid_amt_mean"].replace(0, np.nan)
            df["amt_vs_uid_mean"] = df["TransactionAmt"] / uid_mean

        for column, counts in self.frequency_maps_.items():
            df[f"{column}_freq"] = df[column].map(counts)
        for column, codes in self.category_maps_.items():
            df[f"{column}_code"] = df[column].map(codes).astype("float32")

        return self._drop_non_features(df)

    def _drop_non_features(self, df: pd.DataFrame) -> pd.DataFrame:
        redundant_v = set(find_v_columns(df)) - set(self.v_kept_)
        unencoded_text = {name for name in df.columns if df[name].dtype == object}
        dropped = (
            set(ID_COLUMNS)
            | set(TIME_INDEX_COLUMNS)
            | {TARGET}
            | set(LOW_CARDINALITY)
            | set(HIGH_CARDINALITY_TEXT)
            | set(EMAIL_SUFFIX_COLUMNS)
            | redundant_v
            | unencoded_text
        )
        keep = [name for name in df.columns if name not in dropped]
        return df[keep].astype("float32")
