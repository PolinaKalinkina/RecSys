import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple
import pandas as pd
from my_model import ALSModel
import numpy as np

cfg_data = {
    "user_column": "user_id",
    "item_column": "item_id",
    "date_column": "timestamp",
    "rating_column": "weight",
    "weighted": False,
    "use_gpu": True,
    "dataset_names": ["smm", "zvuk"],
    "data_dir": "./",
    "model_dir": "./saved_models",
}


def create_intersection_dataset(
    smm_events: pd.DataFrame,
    zvuk_events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smm_item_count = smm_events["item_id"].nunique()
    zvuk_item_count = zvuk_events["item_id"].nunique()

    zvuk_events["item_id"] += smm_item_count
    merged_events = pd.concat([smm_events, zvuk_events])
    item_indices_info = pd.DataFrame(
        {"left_bound": [0, smm_item_count],
         "right_bound": [smm_item_count, smm_item_count + zvuk_item_count]},
        index=["smm", "zvuk"]
    )
    user_ids = set(merged_events["user_id"])
    encoder = {id: n for n, id in enumerate(user_ids)}
    merged_events["user_id"] = merged_events["user_id"].map(encoder)
    return merged_events, item_indices_info, encoder


def create_one_dataset(
    events: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    item_count = events["item_id"].nunique()

    # events = pd.concat([smm_events, zvuk_events])
    # item_indices_info = pd.DataFrame(
    #     {"left_bound": [0, smm_item_count],
    #      "right_bound": [smm_item_count, smm_item_count + zvuk_item_count]},
    #     index=["smm", "zvuk"]
    # )
    user_ids = set(events["user_id"])
    encoder = {id: n for n, id in enumerate(user_ids)}

    events["user_id"] = events["user_id"].map(encoder)
    return events, None, encoder

class Trainer:

    def __init__(self, factors=200, regularization=0.005, iterations=200, alpha=20):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

    def fit(self) -> None:
        smm_path = os.path.join(cfg_data["data_dir"], "train_smm.parquet")
        zvuk_path = os.path.join(cfg_data["data_dir"], "train_zvuk.parquet")
        print("Train smm-events:", smm_path)
        print("Train zvuk-events:", zvuk_path)
        smm_events = pd.read_parquet(smm_path)
        zvuk_events = pd.read_parquet(zvuk_path)
        
        train_zvuk_events, indices_info, encoder_zvuk = create_one_dataset(zvuk_events)
        train_smm_events, indices_info, encoder_smm = create_one_dataset(smm_events)
        train_zvuk_events["weight"] = 1
        train_smm_events["weight"] = 1
        
        self.zvuk_model = ALSModel(
            cfg_data,
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
        )
        self.zvuk_model.fit(train_zvuk_events)
        self.zvuk_model.users_encoder = encoder_zvuk

        self.smm_model = ALSModel(
            cfg_data,
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
        )
        self.smm_model.fit(train_smm_events)
        self.smm_model.users_encoder = encoder_smm
        
        # md = Path(cfg_data["model_dir"])
        # md.mkdir(parents=True, exist_ok=True)
        # with open(md / "als_zvuk.pickle", "bw") as f:
        #     pickle.dump(self.zvuk_model, f)
        # indices_info.to_parquet(md / "indices_info.parquet")


    def predict(self,subset_name: str) -> None:
        # with open(Path(cfg_data["model_dir"]) / "als.pickle", "br") as f:
        #     my_model: ALSModel = pickle.load(f)
        
        my_model = self.zvuk_model if subset_name == "zvuk" else self.smm_model
        
        my_model.model = my_model.model #.to_cpu()
        encoder = my_model.users_encoder
        decoder = {n: id for id, n in encoder.items()}

        test_data = pd.read_parquet(os.path.join(cfg_data["data_dir"], f"test_{subset_name}.parquet"))

        test_data["user_id"] = test_data["user_id"].map(encoder)
        
        test_data["weight"] = 1

        recs, user_ids = my_model.recommend_k(test_data, k=10)
        recs = pd.Series(recs.tolist(), index=user_ids)
        recs = recs.reset_index()
        recs.columns = ["user_id", "item_id"]
        recs["user_id"] = recs["user_id"].map(decoder)

        prediction_path = Path(cfg_data["data_dir"]) / f"submission_{subset_name}.parquet"
        recs.to_parquet(prediction_path)


def main():
    trainer = Trainer()
    trainer.fit()
    for subset_name in cfg_data["dataset_names"]:
        trainer.predict(subset_name)


if __name__ == "__main__":
    main()
