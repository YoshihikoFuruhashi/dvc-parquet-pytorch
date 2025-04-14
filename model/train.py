import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import ParquetTextDataset

def main():
    parquet_path = "data/processed/train.parquet"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = ParquetTextDataset(parquet_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        print(input_ids.shape, labels.shape)
        break

if __name__ == "__main__":
    main()
