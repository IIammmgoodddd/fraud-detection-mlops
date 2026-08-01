from pathlib import Path

import pandas as pd


def load_transaction_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    raw_path = Path(raw_dir)
    transactions = pd.read_csv(raw_path / "train_transaction.csv")
    identity = pd.read_csv(raw_path / "train_identity.csv")

    # Left join: only ~24% of transactions have identity data. An inner join
    # would silently drop the other 76% instead of keeping them with null
    # identity columns.
    return transactions.merge(identity, on="TransactionID", how="left")
