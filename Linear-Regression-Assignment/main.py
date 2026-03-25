from src.data_cleaning import load_and_clean
from src.eda import perform_eda
from src.model import train_model
from src.evaluation import evaluate_model

TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data.csv"

def main():
    train_df, test_df = load_and_clean(TRAIN_PATH, TEST_PATH)

    # EDA on training data
    perform_eda(train_df)

    # Train model
    model = train_model(train_df)

    # Evaluate model
    evaluate_model(model, test_df)

if __name__ == "__main__":
    main()
