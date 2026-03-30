"""Download the full TinyStories dataset and save as text files."""
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def main():
    from datasets import load_dataset

    print("Loading full TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    train_stories = ds["train"]["text"]
    val_stories = ds["validation"]["text"]

    print(f"Train: {len(train_stories)} stories")
    print(f"Val: {len(val_stories)} stories")

    # Save train
    train_path = os.path.join(DATA_DIR, "tinystories_train.txt")
    print(f"Writing {train_path}...")
    with open(train_path, "w") as f:
        for story in train_stories:
            f.write(story.strip() + "\n\n")

    # Save val
    val_path = os.path.join(DATA_DIR, "tinystories_val.txt")
    print(f"Writing {val_path}...")
    with open(val_path, "w") as f:
        for story in val_stories:
            f.write(story.strip() + "\n\n")

    # Report sizes
    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)
    print(f"\nTrain: {train_size / 1e6:.0f} MB ({train_size / 1e9:.2f} GB)")
    print(f"Val: {val_size / 1e6:.0f} MB")
    print(f"Total: {(train_size + val_size) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
