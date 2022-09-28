import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf, DictConfig
import hydra

PROJECT_PATH = Path('.').absolute()
DATA_PATH = Path(PROJECT_PATH, 'data', 'titanic')
EXPERIMENT_PATH = Path(PROJECT_PATH, 'experiments', 'titanic')

sys.path.append(str(PROJECT_PATH))
from experiments.titanic.preprocess import TitanicProcess
from base.network import TabTransformerNet


def load_data():
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
    cat_levels = list(inputs_train[proc.cat_cols_out].max(axis=0).astype(np.int32) + 1)
    # drop_col_ind = np.where(np.array(cat_levels)==1, 0, 1)
    inputs_test, labels_test = proc.process(test, train=False)

    num_inputs_train = inputs_train[proc.num_cols_out].values.astype(np.float32)
    cat_inputs_train = inputs_train[proc.cat_cols_out].values.astype(np.float32)
    labels_train = labels_train.values.astype(np.float32)

    num_inputs_test = inputs_test[proc.num_cols_out].values.astype(np.float32)
    cat_inputs_test = inputs_test[proc.cat_cols_out].values.astype(np.float32)
    labels_test = labels_test.values.astype(np.float32)

    return cat_levels, num_inputs_train, cat_inputs_train, labels_train, num_inputs_test, cat_inputs_test, labels_test

@hydra.main(config_path='', config_name='cfg.yaml')
def main(cfg: DictConfig) -> None:
# def main():
#     cfg = OmegaConf.load(Path(TASK_PATH, 'config', 'tab_cfg.yaml'))

    # os.system('mlflow ui --backend-store-uri' +
    #           'file://' + hydra.utils.get_original_cwd() + '/mlruns')
    # http://127.0.0.1:5000/

    # load data
    cat_levels, num_inputs_train, cat_inputs_train, labels_train, num_inputs_test, cat_inputs_test, labels_test\
        = load_data()
    cfg.data.input_column_classes = str(cat_levels)

    # learn TabTransformer
    net = TabTransformerNet(cfg)
    net.learn(
        cat_inputs_train, num_inputs_train, labels_train,
        [[cat_inputs_test, num_inputs_test], labels_test]
        )

    # if you run python console or without hydra, then please set the project_path (since mlflow directory)
    preds = net.model([cat_inputs_test, num_inputs_test])
    preds_binary = np.round(preds)

    print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(labels_test, preds_binary)))
    print('accuracy score {:.4f}'.format(metrics.accuracy_score(labels_test, preds_binary)))
    print('F1 score is: {:.4f}'.format(metrics.f1_score(labels_test, preds_binary)))
    print('AUROC score is: {:.4f}'.format(metrics.roc_auc_score(labels_test, preds)))

    return net

if __name__ == "__main__":
    net = main()

    # cat_levels, num_inputs_train, cat_inputs_train, labels_train, num_inputs_test, cat_inputs_test, labels_test = load_data()
