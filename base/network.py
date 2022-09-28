from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from tensorflow.keras import models, layers
from base.functions import TabTransformer
from base.procedure import NetworkLearningProcedure, BaseNetworkLearn

class Network(object):
    def __init__(self):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.model: models.Model = None
        self.procedure: NetworkLearningProcedure = None
        self.strategy: tf.distribute.MirroredStrategy = None

    def create_model(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

class TabTransformerNet(Network):

    def __init__(self, config: DictConfig, procedure=BaseNetworkLearn, distribute=False):
        super(TabTransformerNet, self).__init__()

        self.config = config

        self.input_column_classes = eval(config.data.input_column_classes)
        self.input_num_size = config.data.input_num_size
        self.num_outputs = config.data.num_outputs

        self.embed_size = config.model.embed_size
        self.identifier_ratio = config.model.identifier_ratio
        self.aggregate_method = config.model.aggregate_method
        self.depth = config.model.depth
        self.num_heads = config.model.num_heads
        self.dropout_rate = config.model.dropout_rate
        self.mlp_unit_ratios = eval(config.model.mlp_unit_ratios)
        self.inference_activation = config.model.inference_activation

        self.random_seed = eval(config.random.random_seed)

        self.model: models.Model = None
        self.strategy: tf.distribute.MirroredStrategy = None

        if not distribute:
            self.strategy = None
            self.model = self.create_model()
        else:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model = self.create_model()

        self.procedure = procedure(self.model, config, self.strategy)

    def create_model(self):
        model = TabTransformer(
            input_column_classes=self.input_column_classes, input_num_size=self.input_num_size,
            embed_size=self.embed_size, identifier_ratio=self.identifier_ratio, aggregate_method=self.aggregate_method,
            depth=self.depth, num_heads=self.num_heads, dropout_rate=self.dropout_rate,
            mlp_unit_ratios=self.mlp_unit_ratios, num_outputs=self.num_outputs,
            inference_activation=self.inference_activation,
            random_seed=self.random_seed
        )
        return model

    def learn(self, cat_inputs, num_inputs, labels, valid_data, verbose=1, project_path=None):
         self.procedure.learn(cat_inputs, num_inputs, labels, valid_data, verbose, project_path)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from pathlib import Path
    import pandas as pd
    import numpy as np
    PROJECT_PATH = Path('.').absolute()
    CONFIG_PATH = Path(PROJECT_PATH, 'config')
    DATA_PATH = Path(PROJECT_PATH, 'data')

    cfg = OmegaConf.load(Path(CONFIG_PATH, 'cfg.yaml'))
    net = TabTransformerNet(cfg)

    # datasets
    df_user = pd.read_csv(Path(DATA_PATH, 'df_users.csv'))
    df_act = pd.read_csv(Path(DATA_PATH, 'df_user_activities.csv'))

    # df_user columns
    id_col = 'user_uuid'
    cat_cols = ['sex', 'age_group', 'os']
    date_col = 'date_joined'
    group_col = 'marketing_channel'

    # df_act columns
    num_cols = ['visits', 'revenue']
    target = 'marketing_channel'

    df = pd.merge(df_user, df_act, how='left', on=id_col)

    df['age_group'] = df['age_group'].fillna('nan')

    cat_voca = {
        "sex": sorted(list(df["sex"].unique())),
        "age_group": sorted(list(df["age_group"].unique())),
        "os": sorted(list(df["os"].unique())),
    }

    cat_voca_value = {}
    for cat in list(cat_voca.keys()):
        cat_voca_value[cat] = np.arange(0, len(cat_voca[cat])).tolist()

    df = df.dropna(how='any')
    cat_df = df[cat_cols]

    for cat_col in cat_cols:
        cat_df[cat_col].replace(to_replace=cat_voca[cat_col], value=cat_voca_value[cat_col], inplace=True)

    cat_inputs = cat_df.values
    num_inputs = df[num_cols].values
    labels = pd.get_dummies(df[target], drop_first=True).values
    pd.get_dummies(df[target], drop_first=True).value_counts()

    net.learn(cat_inputs, num_inputs, labels, None, 1, PROJECT_PATH)