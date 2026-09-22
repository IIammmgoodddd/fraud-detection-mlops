import numpy as np
import pandas as pd
import pytest

from src.features import (
    EMAIL_BOTH_MISSING,
    EMAIL_DIFFERENT_DOMAIN,
    EMAIL_PURCHASER_MISSING,
    EMAIL_RECIPIENT_MISSING,
    EMAIL_SAME_DOMAIN,
    TIME_INDEX_COLUMNS,
    TARGET,
    FeatureEngineer,
    add_email_features,
    add_time_features,
    count_decimal_places,
    find_missingness_blocks,
    select_decorrelated,
    split_by_time,
    validate_required_columns,
)

SECONDS_PER_DAY = 24 * 60 * 60


def make_transactions(days: list[int], **overrides: list) -> pd.DataFrame:
    size = len(days)
    frame = pd.DataFrame(
        {
            "TransactionID": range(size),
            "TransactionDT": [day * SECONDS_PER_DAY + 3600 for day in days],
            "isFraud": [index % 2 for index in range(size)],
            "TransactionAmt": [10.0 + index * 1.25 for index in range(size)],
            "ProductCD": ["W", "C"] * (size // 2) + ["W"] * (size % 2),
            "card1": [1000 + index % 4 for index in range(size)],
            "card4": ["visa", "mastercard"] * (size // 2) + ["visa"] * (size % 2),
            "addr1": [200 + index % 3 for index in range(size)],
            "D1": [float(index % 5) for index in range(size)],
            "D2": [float(index % 7) for index in range(size)],
            "P_emaildomain": ["gmail.com", "web.de"] * (size // 2)
            + ["gmail.com"] * (size % 2),
            "R_emaildomain": ["gmail.com", None] * (size // 2)
            + ["gmail.com"] * (size % 2),
            "DeviceInfo": ["Windows", None] * (size // 2) + ["Windows"] * (size % 2),
            "V1": [float(index) for index in range(size)],
            "V2": [float(index) * 2 for index in range(size)],
            "V3": [float((index * 7) % 5) for index in range(size)],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


@pytest.fixture
def fitted_split() -> tuple[FeatureEngineer, pd.DataFrame, pd.DataFrame]:
    frame = make_transactions([index * 3 for index in range(60)])
    train, val = split_by_time(frame)
    return FeatureEngineer().fit(train), train, val


def test_split_by_time_leaves_a_gap_between_train_and_val():
    frame = make_transactions(list(range(200)))
    train, val = split_by_time(frame, train_end_day=122, gap_days=30)

    train_days = train["TransactionDT"] // SECONDS_PER_DAY
    val_days = val["TransactionDT"] // SECONDS_PER_DAY
    assert train_days.max() <= 122
    assert val_days.min() > 152


def test_transform_matches_fit_schema_on_both_splits(fitted_split):
    engineer, train, val = fitted_split
    train_features = engineer.transform(train)
    val_features = engineer.transform(val)

    assert train_features.columns.equals(val_features.columns)
    assert list(train_features.columns) == engineer.feature_names_
    assert (train_features.dtypes == "float32").all()
    assert len(train_features) == len(train)
    assert len(val_features) == len(val)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="Call fit"):
        FeatureEngineer().transform(make_transactions([1, 2, 3]))


def test_missing_required_column_raises_with_a_clear_message():
    frame = make_transactions([1, 2]).drop(columns=["DeviceInfo"])
    with pytest.raises(KeyError, match="DeviceInfo"):
        validate_required_columns(frame)


def test_email_suffix_codes_come_from_the_fitted_map_not_the_current_split():
    train = make_transactions(
        list(range(20)),
        P_emaildomain=["gmail.com", "web.de", "mail.es", "aol.net"] * 5,
    )
    # This split never sees a .de address, so per-call encoding would renumber .es and .net.
    val = make_transactions(
        list(range(200, 212)),
        P_emaildomain=["gmail.com", "mail.es", "aol.net"] * 4,
    )
    engineer = FeatureEngineer().fit(train)

    train_features = engineer.transform(train)
    val_features = engineer.transform(val)
    code_by_domain = {
        domain: train_features.loc[
            train["P_emaildomain"] == domain, "p_email_suffix_code"
        ].iloc[0]
        for domain in ["gmail.com", "mail.es", "aol.net"]
    }
    for domain, expected_code in code_by_domain.items():
        actual = val_features.loc[val["P_emaildomain"] == domain, "p_email_suffix_code"]
        assert (actual == expected_code).all(), (
            f"{domain} was renumbered between splits"
        )


def test_categories_only_seen_in_val_encode_as_null_rather_than_a_new_code():
    train = make_transactions(list(range(20)))
    val = make_transactions(list(range(200, 208)), card4=["discover"] * 8)
    engineer = FeatureEngineer().fit(train)

    val_features = engineer.transform(val)
    assert val_features["card4_code"].isna().all()


def test_zero_uid_mean_yields_null_rather_than_infinity():
    # One cardholder whose purchase and refund cancel out, so the mean amount is zero.
    train = make_transactions(
        [1, 1],
        card1=[1000, 1000],
        addr1=[200, 200],
        D1=[0.0, 0.0],
        TransactionAmt=[-0.5, 0.5],
    )
    features = FeatureEngineer().fit(train).transform(train)

    assert features["uid_amt_mean"].eq(0).all()
    assert not np.isinf(features["amt_vs_uid_mean"]).any()
    assert features["amt_vs_uid_mean"].isna().all()


def test_fit_is_idempotent_when_called_twice(fitted_split):
    engineer, train, val = fitted_split
    first = engineer.transform(val)
    second = engineer.fit(train).transform(val)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(0.0, 0), (0.5, 1), (0.25, 2), (0.125, 3), (0.1234, 4), (0.9, 1), (np.nan, 0)],
)
def test_count_decimal_places(amount, expected):
    assert count_decimal_places(pd.Series([amount]))[0] == expected


def test_select_decorrelated_drops_duplicates_and_keeps_constants():
    frame = pd.DataFrame(
        {
            "base": [1.0, 2.0, 3.0, 4.0, 5.0],
            "doubled": [2.0, 4.0, 6.0, 8.0, 10.0],
            "unrelated": [5.0, 3.0, 9.0, 1.0, 7.0],
            "constant": [1.0] * 5,
        }
    )
    kept = select_decorrelated(frame, list(frame.columns), threshold=0.9)

    assert kept == ["base", "unrelated", "constant"]


def test_select_decorrelated_keeps_everything_when_no_complete_rows():
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, np.nan]})
    assert select_decorrelated(frame, ["a", "b"]) == ["a", "b"]


def test_select_decorrelated_handles_an_empty_column_list():
    assert select_decorrelated(pd.DataFrame({"a": [1.0]}), []) == []


# --- restored leakage + behaviour coverage ---


def one_entity_across_the_gap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows sharing a single uid, straddling the train/validation gap."""
    days = [0, 10, 20, 200, 210]
    frame = make_transactions(
        days,
        card1=[1000] * 5,
        addr1=[200.0] * 5,
        D1=[float(day) for day in days],  # day - D1 constant -> one entity
    )
    return split_by_time(frame)


def test_split_by_time_discards_the_gap_rows():
    frame = make_transactions(list(range(200)))
    train, val = split_by_time(frame, train_end_day=122, gap_days=30)
    assert len(train) + len(val) < len(frame)


def test_entity_aggregates_are_fitted_on_train_only():
    train, val = one_entity_across_the_gap()
    engineer = FeatureEngineer().fit(train)

    assert (engineer.transform(val)["uid_count"] == len(train)).all()


def test_frequency_encoding_uses_train_counts_and_nulls_unseen_values():
    train, val = one_entity_across_the_gap()
    val = val.copy()
    val.loc[val.index[-1], "card1"] = 9999

    encoded = FeatureEngineer().fit(train).transform(val)["card1_freq"]

    assert encoded.iloc[0] == len(train), "seen values must use the train count"
    assert pd.isna(encoded.iloc[-1]), "unseen values must be null, not self-counted"


def test_target_and_identifiers_never_reach_the_feature_matrix(fitted_split):
    engineer, train, _ = fitted_split
    features = engineer.transform(train)

    for forbidden in (TARGET, "TransactionID", "TransactionDT", "uid"):
        assert forbidden not in features.columns


def test_hour_is_derived_from_transaction_seconds():
    frame = make_transactions([0, 0], TransactionDT=[0, 7 * SECONDS_PER_DAY // 24])
    assert add_time_features(frame)["hour"].tolist() == [0, 7]


def test_email_match_state_distinguishes_missing_from_matching():
    frame = make_transactions(
        [1] * 5,
        P_emaildomain=[None, "a.com", None, "a.com", "a.com"],
        R_emaildomain=[None, None, "b.com", "a.com", "b.com"],
    )
    assert add_email_features(frame)["email_match_state"].tolist() == [
        EMAIL_BOTH_MISSING,
        EMAIL_RECIPIENT_MISSING,
        EMAIL_PURCHASER_MISSING,
        EMAIL_SAME_DOMAIN,
        EMAIL_DIFFERENT_DOMAIN,
    ]


def test_find_missingness_blocks_groups_columns_by_null_count():
    frame = pd.DataFrame({"A": [1.0, None], "B": [1.0, None], "C": [1.0, 2.0]})
    blocks = find_missingness_blocks(frame, ["A", "B", "C"])

    assert sorted(len(block) for block in blocks) == [1, 2]
    assert {"A", "B"} in [set(block) for block in blocks]


def test_raw_time_indexes_are_excluded_from_features(fitted_split):
    """A split on a bare day number cannot generalise past the training window."""
    engineer, train, _ = fitted_split
    features = engineer.transform(train)

    for column in TIME_INDEX_COLUMNS:
        assert column not in features.columns
    assert "hour" in features.columns, "cyclic time features are still wanted"


def test_d9_is_not_normalised_because_it_is_a_fraction_of_a_day():
    train, _ = one_entity_across_the_gap()
    features = FeatureEngineer().fit(train).transform(train)

    assert "D9_norm" not in features.columns
    assert "D2_norm" in features.columns
