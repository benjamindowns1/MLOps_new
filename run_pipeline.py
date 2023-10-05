from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":

    # Run the pipeline
    # print(Client().active_stack.experiment_tracker.get)
    train_pipeline(data_path="E:\Github\MLOps_new\dataset\olist_customers_dataset.csv")