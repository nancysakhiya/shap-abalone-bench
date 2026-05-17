import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

def load_abalone(save_path: str = "data/abalone.csv") -> pd.DataFrame:
    
    Path("data").mkdir(exist_ok=True)

    # Fetch directly from UCI using their official API
    abalone = fetch_ucirepo(id=1)

    X = abalone.data.features.copy()
    y = abalone.data.targets.copy()

    # Encode categorical Sex column
    sex_map = {"M": 0, "F": 1, "I": 2}
    X["Sex"] = X["Sex"].map(sex_map)

    df = pd.concat([X, y], axis=1)
    df.to_csv(save_path, index=False)

    print(f"Dataset saved to {save_path}")
    print(f"Shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nTarget (Rings) stats:\n{df['Rings'].describe()}")
    return df


if __name__ == "__main__":
    df = load_abalone()
    print("\nFirst 5 rows:")
    print(df.head())