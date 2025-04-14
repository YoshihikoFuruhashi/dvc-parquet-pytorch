import pandas as pd
import os

def main():
    input_path = "raw/train.csv"
    output_path = "data/processed/train.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    df.to_parquet(output_path, compression="snappy")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
