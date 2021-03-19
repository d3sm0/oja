import enum

import haiku as hk
import jax
import jax.numpy as jnp


def Print():
    """Layer construction function for an identity layer."""
    init_fun = lambda rng, input_shape: (input_shape, ())

    def apply_fn(params, inputs, **kwargs):
        print("hello")
        return inputs

    return init_fun, apply_fn


class Operation(enum.IntEnum):
    accumulate = 0  # accumulate
    tangent = 1  # forward
    adjoint = 2  # backward


class Project(hk.Module):
    def __init__(self, h_dim=4):
        super(Project, self).__init__()
        self.fc = hk.Sequential([hk.Linear(h_dim),
                                 jax.nn.relu]
                                )

    def __call__(self, x):
        h = self.fc(x)
        return h


class Predict(hk.Module):
    def __init__(self, h_dim=4):
        super(Predict, self).__init__()
        self.fc = hk.Sequential([hk.Linear(h_dim)])

    def __call__(self, x):
        h = self.fc(x).squeeze()
        return h


class Resnet(hk.Module):

    def __init__(self, h_dim=4):
        super(Resnet, self).__init__(name="resnet")
        self.fc = hk.Sequential([hk.Linear(h_dim),
                                 jax.nn.relu,
                                 hk.Linear(h_dim)]
                                )

    def __call__(self, x):
        h = self.fc(x)
        return jax.nn.relu(x + h)


def network(x):
    out = hk.Sequential([
        Project(),
        Resnet(),
        Predict(),
    ])
    return out(x)


def resnet(x):
    return Resnet()(x)


def predict(x):
    return Predict()(x)


def project(x):
    return Project()(x)


def _linearize(f, jacobian):
    f_w = jacobian(f)

    def _get_jacobian(*args):
        jacobian = f_w(*args)
        jacobian, _ = jax.tree_flatten(jacobian)
        out = jnp.hstack([j.reshape(j.shape[0], -1) for j in jacobian])
        return out

    return _get_jacobian


def linearize_fwd(f):
    return _linearize(f, jax.jacfwd)


def linearize_rev(f):
    return _linearize(f, jax.jacrev)


# How to get the gradient to propagate backward?
x = jnp.ones((4,))
key_gen = hk.PRNGSequence(42)
forward = hk.without_apply_rng(hk.transform(network))

resnet = hk.without_apply_rng(hk.transform(resnet))
predict = hk.without_apply_rng(hk.transform(predict))
project = hk.without_apply_rng(hk.transform(project))

resnet_params = resnet.init(next(key_gen), x)
predict_params = predict.init(next(key_gen), x)
project_params = project.init(next(key_gen), x)

h = project.apply(project_params, x)
h1 = resnet.apply(resnet_params, h)
y = predict.apply(predict_params, h1)

# TODO undertand what does it mean to give the basis for each  evaluation
# subgraph -> ve -> accumulation path -> build a jacobian of the subgraph with a.p. and put this inside the chain for the ith module

j_0 = linearize_rev(project.apply)(project_params, x)
j_mid = linearize_rev(resnet.apply)(resnet_params, h)
j_last = linearize_rev(predict.apply)(predict_params, h1)

for method_invocation in hk.experimental.eval_summary(resnet.apply)(h):
    print(method_invocation)



h, vjp_0 = jax.vjp(project.apply, project_params, x)
h1, vjp_1 = jax.vjp(resnet.apply, resnet_params, h)
# forward mode
v = jax.grad(predict.apply, argnums=0)(predict_params, h1)
out_rev = vjp_0(vjp_1(v)[1])[1]

# h, jvp_0 = jax.jvp(project.apply, (project_params, x), tangents=())
y, ds = jax.jvp(predict.apply, (predict_params, h1), vjp_1(v)[1])
# h1, jvp_1 = jax.jvp(resnet.apply, (resnet_params, h), tangents=())

jac = jax.jacfwd(project.apply, argnums=0)(project_params, x)
w = jax.jacfwd(resnet.apply, argnums=0)(resnet_params, h)
w2 = jax.jacfwd(predict.apply, argnums=0)(predict_params, h1)

jnp.vdot(v, w2)
# jax.jvp(project.apply, (project_params, x), (v, ))


# out = jax.jacfwd(project.apply, argnums=(0,))(project_params, x)
# print(out)

# apply = jax.partial(project.apply, x=x)
# jax.jvp(predict.apply, _std_basis(project_params))

# a = jax.jacfwd(project.apply)(project_params, )
# tangent = jax.tree_map(jnp.ones_like, project_params)
# a = jax.jvp(project.apply, (project_params, x), tangents=(tangent, x))
# jacfwd()
# h, jvp_0 = jax.jvp(project.apply, (params, x), (ones_like,))
# h1, jvp_1 = jax.jvp(resnet.apply, (params, h))
# out_fwd = jvp_0(v)[1]
# print(out_fwd, out_rev)

# for very node assoicate  a module and a push operation

print("reverse mode")

# d_p, j_in = rev_0(params, x)
# d_p = params["project/~/linear"]
# d_res, j_res = rev_1(params, h)
# d_res_0 = params["resnet/~/linear"]
# d_res_1 = params["resnet/~/linear_1"]
# d_out, j_out = rev_2(params, h1)
# d_out = params["predict/~/linear"]
# jax.vjp(resnet.apply)
# jacobian_chain = j_out

# rev_out = jax.grad(predict.apply, argnums=(0, 1))
# assert fwd(params, x) == rev(params, x)

# dw = rev(params, x)
# dout, dx = rev_out(params_2, h)
#
# params[0][0] - dx @ dw[0][0]
# params[0][1] - dx @ dw[0][1]
# print(dw, dx)
# params_2[0][0] - dout[0][0]
# params_2[0][1] - dout[0][1]
#
# out, vjp = jax.vjp(forward.apply, params, x)
# out, jvp = jax.jvp(forward.apply, params, x)

## input size
# input_size = 4
# hk.transform()
# model = Resnet(input_size)
# node_shape = count_ops_torch(model, input_size)
# x = torch.ones((input_size,))
# y = model(x)
# y.backward()
#
# print(node_shape)
#
## split the graph in layers
## optimize every layer with VE
## optimize the full chain with DP
#
