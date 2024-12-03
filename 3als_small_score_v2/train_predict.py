import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple
import pandas as pd
from my_model_scores import ALSModel
import numpy as np
from typing import Dict, List

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
    both_users: set = set([])
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smm_events = smm_events.copy()
    zvuk_events = zvuk_events.copy()

    smm_item_count = smm_events["item_id"].nunique()
    zvuk_item_count = zvuk_events["item_id"].nunique()

    zvuk_events["item_id"] += smm_item_count
    merged_events = pd.concat([smm_events, zvuk_events])
    item_indices_info = pd.DataFrame(
        {"left_bound": [0, smm_item_count],
         "right_bound": [smm_item_count, smm_item_count + zvuk_item_count]},
        index=["smm", "zvuk"]
    )

    if len(both_users) > 0:
        merged_events = merged_events[merged_events['user_id'].isin(both_users)]

    user_ids = set(merged_events["user_id"])
    encoder = {id: n for n, id in enumerate(user_ids)}

    merged_events["user_id"] = merged_events["user_id"].map(encoder)
    return merged_events, item_indices_info, encoder


def create_one_dataset(
    events: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_ids = set(events["user_id"])
    encoder = {id: n for n, id in enumerate(user_ids)}

    events["user_id"] = events["user_id"].map(encoder)
    return events, None, encoder


def select_users(
    smm_events: pd.DataFrame,
    zvuk_events: pd.DataFrame
) -> Dict[str, List[int]]:
    
    users_split = dict()
    smm_unique = set(smm_events["user_id"].unique())
    zvuk_unique = set(zvuk_events["user_id"].unique())

    both_users = smm_unique & zvuk_unique
    users_split['both'] = both_users
    users_split['smm'] = smm_unique - both_users
    users_split['zvuk']  = zvuk_unique - both_users

    return users_split


class Trainer:

    def __init__(self,
        factors=200,
        regularization=0.005,
        iterations=200,
        alpha=20,
        factors_big=200,
        regularization_big=0.005,
        iterations_big=200,
        alpha_big=20
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.factors_big = factors_big
        self.regularization_big = regularization_big
        self.iterations_big = iterations_big
        self.alpha_big = alpha_big

    def fit(self) -> None:
        smm_path = os.path.join(cfg_data["data_dir"], "train_smm.parquet")
        zvuk_path = os.path.join(cfg_data["data_dir"], "train_zvuk.parquet")
        print("Train smm-events:", smm_path)
        print("Train zvuk-events:", zvuk_path)
        smm_events = pd.read_parquet(smm_path)
        zvuk_events = pd.read_parquet(zvuk_path)

        self.select_dict = select_users(smm_events, zvuk_events)
        
        train_events, indices_info, encoder = create_intersection_dataset(
            smm_events, 
            zvuk_events,
            self.select_dict['both']
        )
        train_events["weight"] = 1
        
        big_model = ALSModel(
            cfg_data,
            factors=self.factors_big,
            regularization=self.regularization_big,
            iterations=self.iterations_big,
            alpha=self.alpha_big,
        )
        big_model.fit(train_events)
        big_model.users_encoder = encoder

        md = Path(cfg_data["model_dir"])
        md.mkdir(parents=True, exist_ok=True)
        with open(md / "als.pickle", "bw") as f:
            pickle.dump(big_model, f)
        indices_info.to_parquet(md / "indices_info.parquet")
    
        train_zvuk_events, _, encoder_zvuk = create_one_dataset(zvuk_events)
        train_smm_events, _, encoder_smm = create_one_dataset(smm_events)
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
        self.indices_info = indices_info

    def get_top_items_for_all_users(self, item_ids1, scores1, item_ids2, scores2, top_n=10):
        """
        Функция для выбора top_n лучших айтемов без дубликатов для всех пользователей из двух наборов предсказаний.

        :param item_ids1: Матрица с айтемами от первой модели, где каждая строка соответствует пользователю.
        :param scores1: Матрица со скорами от первой модели, где каждая строка соответствует пользователю.
        :param item_ids2: Матрица с айтемами от второй модели, где каждая строка соответствует пользователю.
        :param scores2: Матрица со скорами от второй модели, где каждая строка соответствует пользователю.
        :param top_n: Количество лучших айтемов для выбора (по умолчанию 10).
        :return: Матрица с top_n лучшими айтемами без дубликатов для каждого пользователя.
        """
        num_users = item_ids1.shape[0]
        top_items = []

        for user_idx in range(num_users):
            # Объединение предсказаний для текущего пользователя
            combined_item_ids = np.concatenate((item_ids1[user_idx], item_ids2[user_idx]))
            combined_scores = np.concatenate((scores1[user_idx], scores2[user_idx]))

            # Сортировка по скорам в порядке убывания
            sorted_indices = np.argsort(combined_scores)[::-1]
            sorted_item_ids = combined_item_ids[sorted_indices]
            sorted_scores = combined_scores[sorted_indices]

            # Удаление дубликатов
            seen = set()
            unique_items = []
            for item_id, score in zip(sorted_item_ids, sorted_scores):
                if item_id not in seen:
                    seen.add(item_id)
                    unique_items.append(item_id)

            # Выбор top_n лучших айтемов для текущего пользователя
            top_items.append(np.array(unique_items[:top_n]))

        return np.array(top_items)


    def predict(self, subset_name: str) -> None:
        
        my_model = self.zvuk_model if subset_name == "zvuk" else self.smm_model

        my_model.model = my_model.model
        encoder_local = my_model.users_encoder
        decoder_local = {n: id for id, n in encoder_local.items()}
        indices_info = self.indices_info

        test_data = pd.read_parquet(os.path.join(cfg_data["data_dir"], f"test_{subset_name}.parquet"))

        test_local_only = test_data[test_data["user_id"].isin(self.select_dict[subset_name])]
        test_local_only["user_id"] = test_local_only["user_id"].map(encoder_local)
        
        test_local_only["weight"] = 1

        recs_local, _, user_ids = my_model.recommend_k(test_local_only, k=10)
        recs_local = pd.Series(recs_local.tolist(), index=user_ids)
        recs_local = recs_local.reset_index()
        recs_local.columns = ["user_id", "item_id"]
        recs_local["user_id"] = recs_local["user_id"].map(decoder_local)

        with open(Path(cfg_data["model_dir"]) / "als.pickle", "br") as f:
            big_model: ALSModel = pickle.load(f)

        big_model.model = big_model.model.to_cpu()
        encoder_big = big_model.users_encoder
        decoder_big = {n: id for id, n in encoder_big.items()}

        test_big_only = test_data[test_data["user_id"].isin(self.select_dict['both'])].copy()
        test_big_local = test_data[test_data["user_id"].isin(self.select_dict['both'])].copy()
        test_big_only["user_id"] = test_big_only["user_id"].map(encoder_big)
        test_big_local["user_id"] = test_big_local["user_id"].map(encoder_local)
        
        test_big_only["weight"] = 1
        test_big_local["weight"] = 1

        left_bound, right_bound = (
            indices_info["left_bound"][subset_name],
            indices_info["right_bound"][subset_name],
        )

        big_model.model.item_factors[:left_bound, :] = 0
        big_model.model.item_factors[right_bound:, :] = 0

        recs_big, scores_big, user_ids_big = big_model.recommend_k(test_big_only, k=10)
        recs_local_big, scores_local_big, user_ids_big = my_model.recommend_k(test_big_local, k=10)
        recs_big = recs_big - left_bound #np.array([np.array(x - left_bound) for x in recs_big.tolist()])

        recs_big = self.get_top_items_for_all_users(recs_big, scores_big, recs_local_big, scores_local_big)
        print(recs_big)
        recs_big = pd.Series(recs_big.tolist(), index=user_ids_big)
        recs_big = recs_big.reset_index()
        recs_big.columns = ["user_id", "item_id"]
        recs_big["user_id"] = recs_big["user_id"].map(decoder_big)
        recs = pd.concat([recs_big, recs_local], axis=0).reset_index(drop=True)

        prediction_path = Path(cfg_data["data_dir"]) / f"submission_{subset_name}.parquet"
        recs.to_parquet(prediction_path)


def main():
    trainer = Trainer()
    trainer.fit()
    for subset_name in cfg_data["dataset_names"]:
        trainer.predict(subset_name)


if __name__ == "__main__":
    main()
