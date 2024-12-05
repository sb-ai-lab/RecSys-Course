#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import pendulum

from airflow.decorators import dag, task

import os
import pickle 
import yaml

from rs_datasets import MovieLens

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from replay.splitters import LastNSplitter
from replay.data import (
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureSource,
    FeatureType,
    Dataset,
)
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential.callbacks import (
    ValidationMetricsCallback,
)
from replay.models.nn.sequential.postprocessors import RemoveSeenItems
from replay.data.nn import (
    SequenceTokenizer,
    SequentialDataset,
    TensorFeatureSource,
    TensorSchema,
    TensorFeatureInfo
)
from replay.models.nn.sequential import SasRec
from replay.models.nn.sequential.sasrec import (
    SasRecTrainingDataset,
    SasRecValidationDataset,
)

import mlflow
import pandas as pd
import logging


def prepare_feature_schema(is_ground_truth: bool) -> FeatureSchema:
    # Prepare FeatureSchema required to create Dataset
    base_features = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_hint=FeatureHint.QUERY_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="item_id",
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )
    if is_ground_truth:
        return base_features

    all_features = base_features + FeatureSchema(
        [
            FeatureInfo(
                column="timestamp",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )
    return all_features


# [START dag_decorator_usage]
@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
    dag_id="Sasrec_model_DAG",
)
def example_display_name():

    @task(task_id="preprocess_data")
    def preprocess_data():
        # In the current implementation, the SASRec does not take into account the features of items or users. 
        # They are only used to get a complete list of users and items.
        movielens = MovieLens("100k")
        interactions = movielens.ratings
        user_features = movielens.users
        item_features = movielens.items
        logging.info(interactions)

        # Removing duplicates in the timestamp column without changing the original items order where timestamp is the same
        interactions["timestamp"] = interactions["timestamp"].astype("int64")
        interactions = interactions.sort_values(by="timestamp")
        interactions["timestamp"] = interactions.groupby("user_id").cumcount()


        interactions.to_parquet("volumes/interactions.parquet")
        user_features.to_parquet("volumes/user_features.parquet")
        item_features.to_parquet("volumes/item_features.parquet")

        return True
    
    @task(task_id="split_data_and_prepare_datasets")
    def split_data_and_prepare_datasets(preprocessed_data: bool):
        interactions = pd.read_parquet("volumes/interactions.parquet")
        user_features = pd.read_parquet("volumes/user_features.parquet")
        item_features = pd.read_parquet("volumes/item_features.parquet")
        
        # Split interactions into the train, validation and test datasets using LastNSplitter
        splitter = LastNSplitter(
            N=1,
            divide_column="user_id",
            query_column="user_id",
            strategy="interactions",
        )

        raw_test_events, _ = splitter.split(interactions)
        raw_validation_events, raw_validation_gt = splitter.split(raw_test_events)
        raw_train_events = raw_validation_events

        # Create Dataset for the training stage
        train_dataset = Dataset(
            feature_schema=prepare_feature_schema(is_ground_truth=False),
            interactions=raw_train_events,
            query_features=user_features,
            item_features=item_features,
            check_consistency=True,
            categorical_encoded=False,
        )

        # Create Datasets (events and ground_truth) for the validation stage
        validation_dataset = Dataset(
            feature_schema=prepare_feature_schema(is_ground_truth=False),
            interactions=raw_validation_events,
            query_features=user_features,
            item_features=item_features,
            check_consistency=True,
            categorical_encoded=False,
        )
        validation_gt = Dataset(
            feature_schema=prepare_feature_schema(is_ground_truth=True),
            interactions=raw_validation_gt,
            check_consistency=True,
            categorical_encoded=False,
        )

        '''
            A schema shows the correspondence of columns from the source dataset 
            with the internal representation of tensors inside the model
        '''
        tensor_schema = TensorSchema(
            TensorFeatureInfo(
                name="item_id_seq",
                cardinality=train_dataset.item_count,
                is_seq=True,
                feature_type=FeatureType.CATEGORICAL,
                feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, train_dataset.feature_schema.item_id_column)],
                feature_hint=FeatureHint.ITEM_ID,
            )
        )
        with open('volumes/tensor_schema.pickle', 'wb') as handle:
            pickle.dump(tensor_schema, handle)

        
        tokenizer = SequenceTokenizer(tensor_schema, allow_collect_to_master=True)
        tokenizer.fit(train_dataset)

        sequential_train_dataset = tokenizer.transform(train_dataset)

        sequential_validation_dataset = tokenizer.transform(validation_dataset)
        sequential_validation_gt = tokenizer.transform(validation_gt, [tensor_schema.item_id_feature_name])

        sequential_validation_dataset, sequential_validation_gt = SequentialDataset.keep_common_query_ids(
            sequential_validation_dataset, sequential_validation_gt
        )

        with open('volumes/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)

        with open('volumes/sequential_train_dataset.pickle', 'wb') as handle:
            pickle.dump(sequential_train_dataset, handle)
        with open('volumes/sequential_validation_dataset.pickle', 'wb') as handle:
            pickle.dump(sequential_validation_dataset, handle)
        with open('volumes/sequential_validation_gt.pickle', 'wb') as handle:
            pickle.dump(sequential_validation_gt, handle)

        return train_dataset.item_count



    @task(task_id="model_fit")
    def model_fit(train_dataset_item_count: int) -> str: 
        with open(f'volumes/sequential_validation_dataset.pickle', 'rb') as handle:
            sequential_validation_dataset = pickle.load(handle)
        with open(f'volumes/sequential_validation_gt.pickle', 'rb') as handle:
            sequential_validation_gt = pickle.load(handle)
        with open(f'volumes/tensor_schema.pickle', 'rb') as handle:
            tensor_schema = pickle.load(handle)
        with open(f'volumes/sequential_train_dataset.pickle', 'rb') as handle:
            sequential_train_dataset = pickle.load(handle)

        # Train model
        # Create SASRec model instance and run the training stage using lightning
        '''
            After each epoch validation metrics are shown. You can change the list of validation 
                metrics in ValidationMetricsCallback 
            The model is determined to be the best and is saved if the metric updates its maximum 
                during validation (see the ModelCheckpoint)
        '''
        MAX_SEQ_LEN = 15
        BATCH_SIZE = 512
        NUM_WORKERS = 0

        model = SasRec(
            tensor_schema,
            block_count=2,
            head_count=2,
            max_seq_len=MAX_SEQ_LEN,
            hidden_size=64,
            dropout_rate=0.5,
            optimizer_factory=FatOptimizerFactory(learning_rate=0.001),
        )

        csv_logger = CSVLogger(save_dir="model_logs/train", name="SASRec_example")

        checkpoint_callback = ModelCheckpoint(
            dirpath="model_checkpoints",
            save_top_k=1,
            verbose=True,
            monitor="recall@10",
            mode="max",
        )

        validation_metrics_callback = ValidationMetricsCallback(
            metrics=["map", "ndcg", "recall"],
            ks=[1, 5, 10, 20],
            item_count=train_dataset_item_count,
            postprocessors=[RemoveSeenItems(sequential_validation_dataset)]
        )

        trainer = L.Trainer(
            max_epochs=10,
            callbacks=[checkpoint_callback, validation_metrics_callback],
            logger=csv_logger,
        )

        train_dataloader = DataLoader(
            dataset=SasRecTrainingDataset(
                sequential_train_dataset,
                max_sequence_length=MAX_SEQ_LEN,
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        validation_dataloader = DataLoader(
            dataset=SasRecValidationDataset(
                sequential_validation_dataset,
                sequential_validation_gt,
                sequential_train_dataset,
                max_sequence_length=MAX_SEQ_LEN,
            ),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader,
        )
        return checkpoint_callback.best_model_path
    
    
    @task(task_id="upload_model_to_mlflow")
    def upload_model_to_mlflow(model_path: str):
        logging.info(os.getcwd())
        logging.info(os.listdir())
        from models.loader import model_class_for_inference

        model_name = "sasrec"

        with open(f'volumes/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        best_model = SasRec.load_from_checkpoint(model_path)
        logging.info(best_model)

        mlflow_model = model_class_for_inference(model_name)
        wrapped_model = mlflow_model(best_model, tokenizer)

        with open(mlflow_model.config_path, "r", encoding="utf-8") as file:
            conda_env = yaml.safe_load(file)

        mlflow.set_experiment(model_name)
        with mlflow.start_run(run_name="run_origin", nested=True):
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                registered_model_name=model_name,
                python_model=wrapped_model,
                code_path=mlflow_model.code_path,
                conda_env=conda_env["conda_env"],
            )
        return True


    processed_data = preprocess_data()
    train_item_count = split_data_and_prepare_datasets(processed_data)
    model_path = model_fit(train_item_count)
    upload_model_to_mlflow(model_path)



example_dag = example_display_name()
