import jax.numpy as np
import jax.random as rnd
import haiku as hk
from haiku.nets import MLP


def get_model(input_size, layers, batch_size=10):
    dummy_X = np.zeros((batch_size, input_size))
    # Pretty sure we don't need the batch size thing here
    rng_key = rnd.PRNGKey(0)

    def model(x):
        _model = MLP(output_sizes=layers)
        logit = _model(x)
        return logit

    model = hk.without_apply_rng(hk.transform(model))
    params = model.init(rng_key, dummy_X)
    get_logits = model.apply
    return get_logits, params
