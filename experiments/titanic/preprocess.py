from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

PROJECT_PATH = Path('.').absolute()
DATA_PATH = Path(PROJECT_PATH, 'data', 'titanic')
EXPERIMENT_PATH = Path(PROJECT_PATH, 'experiments', 'titanic')


class TitanicProcess(object):
    def __init__(
        self, data_path: Path , id_cols: List, cat_cols: List, num_cols: List, label_col: str,
        impute_method: str='median', impute_indicator: bool=True, discretize_bins: int=5,
        onehot: bool=False,
        return_as_df: bool=True
    ):
        self.data_path = data_path

        self.id_cols = id_cols
        self.cat_cols_in = cat_cols
        self.num_cols_in = num_cols
        self.label_col = label_col

        self.impute_method = impute_method
        self.impute_indicator = impute_indicator
        self.discretize_bins = discretize_bins

        self.onehot = onehot

        self.return_as_df = return_as_df

        self.numeric_scale = None
        self.numeric_impute = None
        self.numeric_discretizer = None

        self.cat_cols_extended = None
        self.categorical_encode = None

        self.num_cols_out = None
        self.cat_cols_out = None
        self.cols_out = None

    def process(self, df, train=False):
        # if train: df = self._load_df(self.data_path, 'train')
        # else: df = self._load_df(self.data_path, 'test')

        df.reset_index(drop=True, inplace=True) # train test split 시 df index (cat), processed_num index 차이로 intermediate_df (concat) 시 data 증가 방지

        label_df = df[self.label_col]

        num = df[self.num_cols_in]
        cat = df[self.cat_cols_in]

        # 1. numeric process
        processed_num = self.numeric_process(num, train=train) # num+num_null+num_disc or num++num_disc

        # 2. filter only numeric (scaled_numeric)
        self.num_cols_out = self.numeric_scale.cols_out
        scaled_numeric = processed_num[self.num_cols_out]

        # 3. filter numerical null indicator
        if self.impute_indicator: null_numeric = processed_num[self.numeric_impute.null_cols]
        else: null_numeric = None

        # 4. filter discretized facotr
        discretized_numeric = processed_num[self.numeric_discretizer.cols_out]

        # 5. cat process with original cat + discretized num (null indicator는 이미 int and binary이므로 추가하지 않음)
        processed_cat = self.cat_process(
            cat_inputs=pd.concat([cat, discretized_numeric], axis=1), train=train,
            cat_cols=self.cat_cols_in+self.numeric_discretizer.cols_out
            )

        # 6. add null indicator to processed_cat and create final df

        if self.impute_indicator:
            self.cat_cols_out = list(null_numeric.columns) + list(processed_cat.columns)
            df_out = pd.concat([scaled_numeric, null_numeric, processed_cat], axis=1)
        else:
            self.cat_cols_out = list(processed_cat.columns)
            df_out = pd.concat([scaled_numeric, processed_cat], axis=1)


        self.cols_out = list(df_out.columns)

        if self.return_as_df:
            return df_out, label_df
        else:
            return df_out.values, label_df.values

    def cat_process(self, cat_inputs, train=False, cat_cols=None):
        if cat_cols is None:
            cat_cols = self.cat_cols_in

        if train:
            self.categorical_encode = CategoricaEncoder(cat_cols, use_onehot=self.onehot, return_as_df=True)
            self.categorical_encode.train(cat_inputs)

        processed_cat = self.categorical_encode.encode(cat_inputs)
        return processed_cat

    def numeric_process(self, numeric_inputs, train=False, num_cols=None):
        if num_cols is None:
            num_cols = self.num_cols_in
        processed_numeric = []

        # 1 impute
        if train:
            self.numeric_impute = NumericalImputer(
                cols=num_cols, method=self.impute_method, indicator=self.impute_indicator,
                return_as_df=True
            )
            self.numeric_impute.train(numeric_inputs)
        imputed_numeric = self.numeric_impute.impute(numeric_inputs)

        # 2 scale
        if train:
            self.numeric_scale = NumericScaler(num_cols, True)
            self.numeric_scale.train(imputed_numeric[num_cols])
        scaled_inputs = self.numeric_scale.scale(imputed_numeric[num_cols])

        processed_numeric.append(scaled_inputs)
        if self.impute_indicator: processed_numeric.append(imputed_numeric[self.numeric_impute.null_cols])

        # 3 discretize  -- 2번이 아닌, 1 번 기준으로 생성
        if train:
            self.numeric_discretizer = NumericalDiscretizer(
                cols=num_cols, bins=self.discretize_bins, return_as_df=True
            )
            self.numeric_discretizer.train(values=imputed_numeric[num_cols])

        disc_numeric = self.numeric_discretizer.discretize(imputed_numeric[num_cols])
        processed_numeric.append(disc_numeric)
        processed_numeric = pd.concat(processed_numeric, axis=1)
        # if self.return_as_df:
        #     processed_numeric = pd.concat(processed_numeric, axis=1)
        # else:
        #     processed_numeric = np.concatenate([df.values for df in processed_numeric], axis=1)
        return processed_numeric

    def _get_labels(self, df):
        return df[self.label_col]

    def _load_df(self, data_dir, type='train'):

        self.df = pd.read_csv(Path(data_dir, f'{type}.csv'))
        return self.df

class NumericalImputer(object):
    def __init__(self, cols: List, method='median', indicator: bool=True, return_as_df: bool=True):
        self.method=method
        self.cols_in = cols
        self.indicator = indicator
        self.return_as_df = return_as_df

        self.null_cols = None
        self.cols_out = None
        self.fill_values = None

    def train(self, values: pd.DataFrame or np.ndarray):
        if self.method == 'median':
            self.fill_values = np.nanmedian(values, axis=0)
        if self.method == 'mean':
            self.fill_values = np.nanmean(values, axis=0)

        if self.indicator:
            self.null_cols = [f'{col}_null'for col in self.cols_in]
            self.cols_out = self.cols_in+self.null_cols
        else:
            self.cols_out = self.cols_in

    def impute(self, values: pd.DataFrame or np.ndarray):
        assert self.fill_values is not None, "call 'self.train' method with the train dataset before impute"

        null_indicator = self._indicator(values)
        imputed_values = np.where(null_indicator, self.fill_values, values)

        if self.indicator:
            imputed_values = np.concatenate([imputed_values, null_indicator], axis=1)

        if self.return_as_df:
            imputed_values = pd.DataFrame(data=imputed_values, columns=self.cols_out)

        return imputed_values

    def _indicator(self, values):
        null_indicator = np.isnan(values).astype(np.int32)
        return null_indicator

class NumericalDiscretizer(object):
    def __init__(self, cols: List, bins: int=5, return_as_df: bool=True):
        self.cols_in = cols
        self.bins = bins
        self.return_as_df = return_as_df


        self.uniform_discrete = None
        self.quantile_discrete = None

        self.uniform_cols = None
        self.quantile_cols = None
        self.cols_out = None

    def train(self, values: pd.DataFrame or np.ndarray):
        # self.uniform_discrete = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform')
        self.quantile_discrete = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='quantile')

        # self.uniform_discrete.fit(values)
        self.quantile_discrete.fit(values)

    def discretize(self, values: pd.DataFrame or np.ndarray):
        # assert (self.uniform_discrete or self.quantile_discrete) is not None, "call 'self.train' method with the train dataset before discretize"
        assert (self.quantile_discrete) is not None, "call 'self.train' method with the train dataset before discretize"
        # num_uniform = self.uniform_discrete.transform(values)
        num_quantile = self.quantile_discrete.transform(values)

        # self.uniform_cols = [f'{col}_uniform_disc'for col in self.cols_in]
        self.quantile_cols = [f'{col}_quantize'for col in self.cols_in]
        # self.cols_out = self.uniform_cols + self.quantile_cols
        self.cols_out = self.quantile_cols

        if self.return_as_df:
            # disc_values = pd.DataFrame(
            #     data=np.concatenate([num_uniform, num_quantile], axis=1),
            #     columns=self.uniform_cols+self.quantile_cols
            # )
            disc_values = pd.DataFrame(
                data=num_quantile,
                columns=self.cols_out
            )
        else:
            # disc_values = np.concatenate([num_uniform, num_quantile], axis=1)
            disc_values = num_quantile
        return disc_values

class NumericScaler(object):
    def __init__(self, cols: List, return_as_df: bool=True):
        self.cols_in = cols
        self.return_as_df = return_as_df

        self.mm_scaler = None  # min max scale
        self.qut_scaler = None  # quantile scale
        self.log_scaler = None
        self.cols_out = None

    def train(self, values: pd.DataFrame or np.ndarray):
        self.mm_scaler = MinMaxScaler()
        self.mm_scaler.fit(values)

        self.qut_scaler = RobustScaler()
        self.qut_scaler.fit(values)

        self.log_scaler = np.log1p
    def scale(self, values: pd.DataFrame or np.ndarray):
        assert self.mm_scaler is not None, "call 'self.train' method with the train dataset before discretize"

        # normalization
        numeric_mm = self.mm_scaler.transform(values)
        columns_mm = [f'{col}_minmax' for col in self.cols_in]

        # quantile
        numeric_qut = self.qut_scaler.transform(values)
        columns_qut = [f'{col}_quantile' for col in self.cols_in]

        # log
        numeric_log = self.log_scaler(values)
        columns_log = [f'{col}_log' for col in self.cols_in]

        self.cols_out = columns_mm+columns_qut+columns_log

        if self.return_as_df:
            return pd.DataFrame(
                data=np.concatenate([numeric_mm, numeric_qut, numeric_log], axis=1), columns=self.cols_out
            )
        else:
            return np.concatenate([numeric_mm, numeric_qut, numeric_log], axis=1)

class CategoricaEncoder(object):
    def __init__(self, cols: List, use_onehot=False, return_as_df=True, categories=None):
        self.cols_in = cols
        self.return_as_df = return_as_df
        self.use_onehot = use_onehot
        if categories: self.categories=categories
        else: self.categories='auto'
        self.cat_encoder = None
        self.cols_out = None

    def train(self, values: pd.DataFrame or np.ndarray):
        if self.use_onehot:
            self.cat_encoder = OneHotEncoder(drop='if_binary', sparse=False)
            self.cat_encoder.fit(values)
            self.cols_out = self.cat_encoder.get_feature_names(self.cols_in)
        else:
            self.cat_encoder = OrdinalEncoder()
            self.cat_encoder.fit(values)
            self.cols_out = self.cols_in

    def encode(self, values: pd.DataFrame or np.ndarray):
        assert self.cat_encoder is not None, "call 'self.train' method with the train dataset before discretize"

        encoded_values = self.cat_encoder.transform(values)

        if self.return_as_df:
            if self.use_onehot: encoded_values = pd.DataFrame(data=encoded_values, columns=self.cols_out)
            else: encoded_values = pd.DataFrame(data=encoded_values, columns=self.cols_out)
        return encoded_values


if __name__ == "__main__":
    train = pd.read_csv(Path(DATA_PATH, 'train.csv'))
    train.Embarked = train.Embarked.fillna('nan')
    train, test = train_test_split(train, test_size=0.3, random_state=42)

    id_cols = ['PassengerId', 'Name']
    label_col = 'Survived'
    cat_cols = ['Sex', 'Pclass', 'Embarked']
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    proc = TitanicProcess(
        data_path=DATA_PATH, id_cols=id_cols, cat_cols=cat_cols, num_cols=num_cols, label_col=label_col,
        impute_method='median', impute_indicator=False, discretize_bins=5, onehot=False,
    )

    inputs_train, labels_train = proc.process(train, train=True)
    cat_levels = list(inputs_train[proc.cat_cols_out].max(axis=0) + 1)
    # drop_col_ind = np.where(np.array(cat_levels)==1, 0, 1)
    inputs_test, labels_test = proc.process(test, train=False)

    num_inputs_train = inputs_train[proc.num_cols_out].values.astype(np.float32)
    cat_inputs_train = inputs_train[proc.cat_cols_out].values.astype(np.float32)
    labels_train = labels_train.values.astype(np.float32)

    num_inputs_test = inputs_test[proc.num_cols_out].values.astype(np.float32)
    cat_inputs_test = inputs_test[proc.cat_cols_out].values.astype(np.float32)
    labels_test = labels_test.values.astype(np.float32)

    num_inputs_train.shape

