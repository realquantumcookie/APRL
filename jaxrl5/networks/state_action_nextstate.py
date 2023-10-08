import flax.linen as nn
import jax.numpy as jnp
# from jax.nn import tanh

# from jaxrl5.networks import default_init
my_init = nn.initializers.xavier_uniform

class StateActionNextState(nn.Module):
    base_cls: nn.Module
    obs_dim: int
    
    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)
        
        residual = nn.Dense(self.obs_dim, kernel_init=my_init())(outputs)
        next_state = observations + residual

        return next_state
