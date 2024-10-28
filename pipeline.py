import numpy as np
import scipy as sp
import polars as pl
import xgboost as xgb
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import rfflearn.cpu as rfflearn
import pickle

class PipeLine:
    def __init__(self,
                 path_to_triggers,
                 path_to_actions,
                 time_splits: int = 10):
        self.df_triggers = pl.read_csv(path_to_triggers)
        self.df_actions = pl.read_csv(path_to_actions)
        self.replacement_dict = {1: 0.5,
                                 2: 1,
                                 3: 0.25}
        self.time_splits = time_splits
        self.built_matrix = None
        self.valid_matrix = None
        self.target_train = None
        self.target_valid = None
        self.model = None
        self._preprocess_data()
        print('Data preprocessing complete. Starting training...')
        self._training_process()


    def _preprocess_data(self):
        self.df_triggers = self.df_triggers.with_columns(pl.col('trigger').cast(pl.Int32),
                                                         pl.col('type').cast(pl.Int32),
                                                         pl.col('date').str.strptime(pl.Datetime,
                                                                                     format='%Y-%m-%d %H:%M:%S'))
        self.df_actions = self.df_actions.with_columns(pl.col('result').cast(pl.Int32),
                                                       pl.col('date').str.strptime(pl.Datetime,
                                                                                   format='%Y-%m-%d %H:%M:%S'))
        self.df_triggers = self.df_triggers.sort('date')
        self.df_actions = self.df_actions.sort('date')


    def _training_process(self):
        split_size = int(31 / self.time_splits)
        i = split_size
        self.df_triggers_train = self.df_triggers.filter(pl.col('date').dt.day() < i)
        self.df_actions_train = self.df_actions.filter(pl.col('date').dt.day() < i)
        self.df_triggers_valid = self.df_triggers.filter((pl.col('date').dt.day() >= i) &
                                                         (pl.col('date').dt.day() <= i))
        self.df_actions_valid = self.df_actions.filter((pl.col('date').dt.day() >= i) &
                                                         (pl.col('date').dt.day() <= i))
        while i <= 30:
            self._build_train(self.df_triggers_train, self.df_actions_train, i)
            self._build_valid(self.df_triggers_valid, self.df_actions_valid, i)
            scaler = StandardScaler()
            scaler.fit_transform(self.built_matrix)
            scaler.transform(self.valid_matrix)
            if not self.model:
                self.model = xgb.XGBClassifier(scale_pos_weight=100)
                X_train, X_test, y_train, y_test = train_test_split(self.built_matrix, self.target_train,
                                        stratify=self.target_train, random_state=42)
                self.model.fit(X_train, y_train)
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.built_matrix, self.target_train,
                                        stratify=self.target_train, random_state=42)
                self.model.fit(X_train, y_train, xgb_model=self.model)
            prediction = self.model.predict(self.valid_matrix)
            print("Current metrics:")
            print("Accuracy: {}".format(accuracy_score(self.target_valid, prediction)))
            print("Precision: {}".format(precision_score(self.target_valid, prediction)))
            print('F1-Score: '.format(f1_score(self.target_valid, prediction)))
            i += split_size
            self.df_triggers_train = self.df_triggers_train.filter(pl.col('date').dt.day() < i)
            self.df_actions_train = self.df_actions.filter(pl.col('date').dt.day() < i)
            self.df_triggers_valid = self.df_triggers.filter((pl.col('date').dt.day() >= i) &
                                                             (pl.col('date').dt.day() <= i))
            self.df_actions_valid = self.df_actions.filter((pl.col('date').dt.day() >= i) &
                                                           (pl.col('date').dt.day() <= i))
        model_file = 'model_pickle.pkl'
        pickle.dump(self.model, open(model_file, 'wb'))

    def _time_split(self, train_set, time_split_value):
        pass  # todo


    def _build_train(self, df_triggers, df_actions, timespan):
        # join both the triggers df and actions and fill missing values (target) with 0
        self.built_matrix = df_triggers.join(df_actions,
                                             on='guid',
                                             how='left').with_columns(pl.col('result').fill_nan(0),
                                                                      pl.col('type'))
        built_matrix_new = (
            self.built_matrix.group_by("guid")
            .agg(
                pl.col("trigger").n_unique().alias("unique_triggers"),
                (pl.col("date").max() - pl.col("date").min())
                .dt.total_days()
                .alias("time_passed"),
                pl.col("type"),
            )
            .with_columns(
                num_triggers=pl.col("type").list.len(),
                trig_type_1=pl.when(pl.col("type").list.contains(1)).then(1).otherwise(0),
                trig_type_2=pl.when(pl.col("type").list.contains(2)).then(1).otherwise(0),
                trig_type_3=pl.when(pl.col("type").list.contains(3)).then(1).otherwise(0),
            )
        )
        self.built_matrix = built_matrix_new.join(
            df_actions, on="guid", how="left"
        ).with_columns(pl.col("result").fill_null(0))

        self.target_train = self.built_matrix['result']

        self.built_matrix = self.built_matrix.drop(['result', 'type', 'date', 'guid'])

        print(self.built_matrix.shape, self.target_train.shape)

    def _build_valid(self, df_triggers, df_actions, timespan):
        self.valid_matrix = df_triggers.join(df_actions,
                                             on='guid',
                                             how='left').with_columns(pl.col('result').fill_nan(0),
                                                                      pl.col('type'))
        built_matrix_new = (
            self.valid_matrix.group_by("guid")
            .agg(
                pl.col("trigger").n_unique().alias("unique_triggers"),
                (pl.col("date").max() - pl.col("date").min())
                .dt.total_days()
                .alias("time_passed"),
                pl.col("type"),
            )
            .with_columns(
                num_triggers=pl.col("type").list.len(),
                trig_type_1=pl.when(pl.col("type").list.contains(1)).then(1).otherwise(0),
                trig_type_2=pl.when(pl.col("type").list.contains(2)).then(1).otherwise(0),
                trig_type_3=pl.when(pl.col("type").list.contains(3)).then(1).otherwise(0),
            )
        )
        self.valid_matrix = built_matrix_new.join(
            df_actions, on="guid", how="left"
        ).with_columns(pl.col("result").fill_null(0))

        self.target_valid = self.valid_matrix['result']

        self.valid_matrix = self.valid_matrix.drop(['result', 'type', 'date', 'guid'])

if __name__ == "__main__":
    print('Insert path to triggers dataset: ')
    path_to_trig = input()
    print('----------------------------------------------------------------------------------')
    print('Insert path to actions dataset: ')
    path_to_act = input()
    print('----------------------------------------------------------------------------------')
    print('Choose how many time splits to perform (recommended - 10): ')
    time_split = int(input())

    training = PipeLine(
        path_to_trig,
        path_to_act,
        time_split
    )
    print('Training complete, saving the model')
