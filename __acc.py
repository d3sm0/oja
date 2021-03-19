# Accumulation path
# ['ResidualBlock/52', 'ResidualBlock/input', 'ResidualBlock/input.3', 'ResidualBlock/Conv2d[conv2]/out', 'ResidualBlock/Conv2d[conv1]/input.2']

import jax
import jax.numpy as jnp


def dense(x, w):
    return jnp.dot(x, w)


def act(x):
    return x


def sum(x, y):
    return x + y


def forward(params, x):
    w_0, w_1 = params
    h = dense(x, w_0)
    h = act(h)
    h = dense(h, w_1)
    h = sum(x, h)
    y = act(h)
    return y


x = jnp.ones(shape=(1,))
w_0 = jnp.ones(shape=(1, 1)) * (-0.0075)
w_1 = jnp.ones(shape=(1, 1)) * 0.5364

y = forward(params=(w_0, w_1), x=x)

out = jax.jacrev(forward, argnums=0)((w_0, w_1), x)
print(out)
h = dense(x, w_0)
dx, dw = jax.jacfwd(dense, argnums=(0, 1))(x, w_0)
h = act(h)
dh = jax.jacfwd(act, argnums=0)(h)
h = dense(h, w_1)
dx, dw = jax.jacfwd(dense, argnums=(0, 1))(h, w_1)
h = sum(x, h)
dx, dw = jax.jacfwd(sum, argnums=(0, 1))(x, h)
y = act(h)
dy = jax.jacfwd(act)(y)
