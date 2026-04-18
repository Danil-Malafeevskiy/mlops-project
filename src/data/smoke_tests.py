import pandas as pd

from scipy.io import arff

def main():
    data, _ = arff.loadarff("data/raw/dataset")
    df = pd.DataFrame(data)
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Class distribution:")
    print(df["Class"].value_counts())

if __name__ == "__main__":
    main()