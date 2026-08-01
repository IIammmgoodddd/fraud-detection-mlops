import pandas as pd

from src.data_loading import load_transaction_data


def test_load_transaction_data_keeps_all_transactions_via_left_join(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["mobile", "desktop"],
        }
    )
    transactions.to_csv(raw_dir / "train_transaction.csv", index=False)
    identity.to_csv(raw_dir / "train_identity.csv", index=False)

    merged = load_transaction_data(raw_dir=str(raw_dir))

    assert len(merged) == len(transactions)
    assert merged["isFraud"].tolist() == [0, 1, 0]
    assert merged.loc[merged["TransactionID"] == 2, "DeviceType"].isna().all()
