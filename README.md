# TabTransformer-Tensorflow: Tensorflow reimplementation of TabTransformer

[TabTransformer is a transformer architecture for tabular dataset introduced by Xin Huang](https://arxiv.org/abs/2012.06678)
that generate robust contextual embedding through self-attention based transformer, column embedding, and unique identifier.

### Improvements

- Simple modules with Tensorflow backend
- Embedding modules
  - Some replications have not been properly implemented for unique identifiers and column embeddings of the original paper.
  - $ e_{\pi}_{i} = [c_{\pi}_{i}, w_{\pi}_{i,j}] $
  - $c_{\pi}_{i}$ is the unique identifier and $w_{\pi}_{i,j}$ is feature-value specific embedding
  - These two vectors could be concatenating and element-wise adding 
- Preprocessing module
  - Section feature engineering at Appendix of original paper
  - numerical features: discretization(quantization), 3 re-scaling method (normalization, quantile, log)

### Enviorments

```
pip install -r requirements.txt
```

### Run

```
python experiments/titanic/train.py
```


### Arguments

The `config/cfg.yaml` has multiple arguments to create and build model.

#### data
- `input_column_classes`: The list of levels of the categorical features
- `input_num_size`: The number of numerical features
- `num_outputs`: The dimension of outputs

#### model
- `name`: model name
- `embed_size`: The dimension of contextual embedding
- `identifier_ratio`: The dimension of unique identifier in contextual embedding. e.g.: 8 : 1/8 of contextual embedding 
- `aggregate_method`: The method to aggregate unique identifier and feature-value specific embedding. 'add', 'concat'
- `depth`: The number of Transformer layers
- `num_heads`: The number of self-attention head
- `dropout_rate`: The ratio of dropout
- `mlp_unit_ratios`: The ratio between the size of contextual embedding and the untis of final fully connected layer. e.g. 2: 2*embed_size
- `inference_activation`: The activation function of output layer. 'sigmoid', 'sofmax', 'linear'

#### train

- `epochs`: The number of epochs to train
- `batch_size`: The size of batch
- `learning_rate`: learning rate to optimization
- `weight_decay`: weight decay 
- `early_stopping`: Bool indicator to implement early stopping
- `stop patience`: The number of patience for early stopping
- `loss_fn`': The tensorflow loss function
- `optimize_fn`: The tensorflow optimization function
- `loss_metric`: The tensorflow metric function for loss
- `evaluate_metric`: The tensorflow metric function for evaluate. e.g. metrics.BinaryAccuracy 

#### random
- `random_seed`: The seed number for randomness

#### mlflow
- `experiment_name`: The experiment name in mlflow

### TODO
- implementation of semi-/self-supervised learning
### References

1. Huang, Xin, et al. "Tabtransformer: Tabular data modeling using contextual embeddings." arXiv preprint arXiv:2012.06678 (2020).
