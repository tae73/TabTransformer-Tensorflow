from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers

class ColumnEmbedding(layers.Layer):
    def __init__(self, column_classes, d, identifier_ratio=4, aggregate_method='concat', random_seed=None):
        """

        :param column_classes: number of classess for each columns
        :param d: final embed size
        :param identifier_ratio: identifier size l = (d / 'identifier_ratio') when aggregate_method='concat', e.g. 2, 4, 8
        :param aggregate_method: 'concat', 'add'
        :param random_seed: random seed
        """
        super().__init__()
        self.method = aggregate_method

        if self.method=='concat':
            self.l = round(d / identifier_ratio) # identifier size
            att_size =np.round(d-self.l)
        else:
            self.l = d
            att_size = d
        self.column_embedding_layers = [
            layers.Embedding(
                    input_dim=classes, output_dim=att_size,
                    embeddings_initializer=initializers.GlorotUniform(seed=random_seed)
                ) for classes in column_classes
        ]

        self.num_columns = len(column_classes)
        self.column_indices = tf.range(start=0, limit=self.num_columns, delta=1)
        self.column_identifiers = layers.Embedding(
                    input_dim=self.num_columns, output_dim=self.l,
                    embeddings_initializer=initializers.GlorotUniform(seed=random_seed)
                )

    def call(self, inputs):
        column_embeddings = []
        for layer, column in zip(self.column_embedding_layers, tf.unstack(inputs, axis=-1)):
            column_embeddings.append(
                layer(column)
            )
        column_embeddings = tf.stack(column_embeddings, axis=1)

        unique_identifiers = self.column_identifiers(self.column_indices)

        if self.method == 'concat':
            unique_identifiers_broad = tf.broadcast_to(
                unique_identifiers, shape=[tf.shape(column_embeddings)[0], self.num_columns, self.l])
            return tf.concat([unique_identifiers_broad, column_embeddings], axis=2)
        else:
            return column_embeddings + unique_identifiers

class MLP(layers.Layer):
    def __init__(self, units: List, dropout_rate: float=0.1, random_seed: int=None):
        super(MLP, self).__init__()
        self.denses = [
            layers.Dense(units=num_unit, activation='selu',
            kernel_initializer=initializers.glorot_uniform(seed=random_seed),
            bias_initializer='zeros') for num_unit in units
        ]
        # self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
            # outputs = self.dropout(outputs, training=training)
        return outputs

class Transformer(layers.Layer):
    def __init__(
            self, embed_size: int, num_heads: int, dropout_rate: float=0.01,
                 random_seed: int=None
    ):
        """

        :param embed_size: the input size of transformer and embedding size of categorical features
        :param num_heads: the number of heads for multi-head attention
        :param expand_size: the dimension of first feed-forward network. 4 * embed_size at original paper
        """

        # The ﬁrst layer expands the embedding to four times its size
        # and the second layer projects it back to its original size.
        super(Transformer, self).__init__()
        expand_size = embed_size * 4

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_size, kernel_initializer=initializers.he_normal(seed=random_seed),
            bias_initializer='zeros'
        )
        self.layernorm_att = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_att = layers.Dropout(dropout_rate)

        self.feedforward_1 = layers.Dense(
            expand_size, activation="relu", kernel_initializer=initializers.he_normal(seed=random_seed),
            bias_initializer='zeros'
        )
        self.feedforward_2 = layers.Dense(
            embed_size, kernel_initializer=initializers.he_normal(seed=random_seed), bias_initializer='zeros'
        )  # TODO activation 없는게 맞나?
        self.layernorm_ffn = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_ffn = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        residual_att = inputs
        outputs = self.attention(inputs, inputs)
        outputs = self.dropout_att(outputs, training=training)
        outputs = self.layernorm_att(residual_att+outputs) # residual

        residual_ffn = outputs
        outputs = self.feedforward_1(outputs)
        outputs = self.feedforward_2(outputs)
        outputs = self.dropout_ffn(outputs, training=training)

        return self.layernorm_ffn(residual_ffn+outputs)

class TabTransformer(models.Model):
    def __init__(
            self, input_column_classes: List, input_num_size: int,
            embed_size, identifier_ratio, aggregate_method,
            depth, num_heads, dropout_rate, mlp_unit_ratios,
            num_outputs, inference_activation='sigmoid',
            random_seed=None
    ):
        super(TabTransformer, self).__init__()

        # embedding of each categorical feature
        self.embedding = ColumnEmbedding(
            input_column_classes, embed_size, identifier_ratio, aggregate_method, random_seed
        )

        # layer normalization for numerical features
        self.numerical_ln = layers.LayerNormalization(epsilon=1e-6)

        # transformers
        self.transformers = [
            Transformer(embed_size, num_heads, dropout_rate, random_seed) for level in range(depth)
        ]

        self.concat = layers.Concatenate()
        mlp_units = [(embed_size*len(input_column_classes)+input_num_size)*factor for factor in mlp_unit_ratios]
        self.mlp = MLP(mlp_units, random_seed)

        self.inference = layers.Dense(units=num_outputs, activation=inference_activation)

    def call(self, inputs: List, training=False):
        """
        :param inputs: inputs[0]: numerical inputs, inputs[1]: list of categorical features of inputs
        :return:
        """
        cat_inputs = inputs[0]
        num_inputs = inputs[1]

        num_layernorm = self.numerical_ln(num_inputs)
        column_embedding = self.embedding(cat_inputs)

        contextual_embedding = column_embedding
        for transformer in self.transformers:
            contextual_embedding = transformer(contextual_embedding, training=training)
        contextual_embedding = layers.Flatten()(contextual_embedding)

        concat = self.concat([contextual_embedding, num_layernorm])
        outputs = self.mlp(concat, training=training)

        return self.inference(outputs)


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    import numpy as np

    PROJECT_PATH = Path('.').absolute()
    CONFIG_PATH = Path(PROJECT_PATH, 'config')
    DATA_PATH = Path(PROJECT_PATH, 'data')

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

    label_voca ={
        "marketing_channel": sorted(list(df["marketing_channel"].unique())),
    }

    cat_voca_value = {}
    for cat in list(cat_voca.keys()):
        cat_voca_value[cat] = np.arange(0, len(cat_voca[cat])).tolist()
    print(cat_voca)
    print(cat_voca_value)

    df = df.dropna(how='any')
    cat_df = df[cat_cols]

    for cat_col in cat_cols:
        cat_df[cat_col].replace(to_replace=cat_voca[cat_col], value=cat_voca_value[cat_col], inplace=True)

    cat_inputs = cat_df.values
    num_inputs = df[num_cols].values

    model = TabTransformer(
        input_column_classes=[2, 6, 2], input_num_size=2,
        embed_size=32, identifier_ratio=4, aggregate_method='concat',
        depth=6, num_heads=8, dropout_rate=0.01,
        mlp_unit_ratios=[4], num_outputs=1, inference_activation='sigmoid',
        random_seed=None
    )

    print(model([cat_inputs, num_inputs], training=False))
