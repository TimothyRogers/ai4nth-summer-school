# %%
# For fixing type hinting - FIXED in python 3.14 but too restrictive dependency!
from __future__ import annotations  # noqa: F404

# We are going to use JAX for our computation, it will make many things much easier
import jax
from jax import config, vmap
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.linalg import solve_triangular

import equinox as eqx
import optax

from matplotlib import pyplot as plt

import numpy as np

from typing import Tuple, Callable, Union, Optional, Any
from functools import partial


# We will want double precision floats for matrix inversions
config.update("jax_enable_x64", True)

# We will need a "nugget" to support positive definiteness
NUGGET = 1e-9


# %%
class MultivariateGaussian(eqx.Module):
    """Multivariate Gaussian distribution with a full covariance matrix."""

    mu: jnp.ndarray  # Shape: (N,)
    sigma: jnp.ndarray  # Shape: (N, N)

    @property
    def N(self) -> int:
        """Returns the dimensionality of the distribution."""
        return self.mu.shape[0]

    @property
    def cholesky(self) -> jnp.ndarray:
        """Computes the lower Cholesky decomposition of the covariance matrix."""
        return jnp.linalg.cholesky(self.sigma + NUGGET * jnp.identity(self.N))

    def sample(self, key: jr.PRNGKey, nsamp: int = 1) -> Tuple[jr.PRNGKey, jnp.ndarray]:
        """Generates random samples from the distribution.

        Args:
            key: JAX PRNG key.
            nsamp: Number of samples to generate.

        Returns:
            A tuple of (next_key, samples) where samples has shape (nsamp, N).
        """
        key, subkey = jr.split(key)
        L = self.cholesky

        z = jr.normal(subkey, (self.N, nsamp))
        ys = self.mu[None, :] + (L @ z).T
        return key, ys

    def loglik(self, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the log-likelihood of data point(s) y.

        Args:
            y: Array of shape (N,) or (M, N) containing data points.

        Returns:
            Log-likelihood values.
        """
        L = self.cholesky
        r = jnp.atleast_2d(y) - jnp.atleast_2d(self.mu)
        R = (solve_triangular(L, r.T, lower=True) ** 2).sum(0)

        return -self.N * jnp.log(2 * jnp.pi) / 2 - jnp.log(L.diagonal()).sum() - 0.5 * R


# %% [markdown]
# In quite a few cases we will be able to constrain ourselves to a situation where we either have a Gaussian distribution where each dimension of the vector is independent, or we only care about the marginal distribution on each dimension. We will see this when we want to plot the results of a GP, we don't need to compute the cross covariances between prediction points which can save a signification amount of computational effort-making the calculation for the variance $\mathcal{O}(N)$ rather than $\mathcal{O}(N^2)$ but losing the cross correlation information.
#
# You will see that we handle this in a very similar way to the full Multivariate Gaussian but we can be even more efficient since we can exploit the diagonal structure of the covariance when computing the log likelihood.


# %%
class DiagonalGaussian(eqx.Module):
    """Multivariate Gaussian distribution with a diagonal covariance matrix.

    `sigma` is a 1D array representing variances for each dimension.
    """

    mu: jnp.ndarray  # Shape: (N,)
    sigma: jnp.ndarray  # Shape: (N,) - Variances

    @property
    def N(self) -> int:
        """Returns the dimensionality of the distribution."""
        return self.mu.shape[0]

    def sample(self, key: jr.PRNGKey, nsamp: int = 1) -> Tuple[jr.PRNGKey, jnp.ndarray]:
        """Generates random samples from the distribution."""
        key, subkey = jr.split(key)
        ys = self.mu[None, :] + jr.normal(subkey, (nsamp, self.N)) * jnp.sqrt(
            self.sigma[None, :]
        )
        return key, ys

    def loglik(self, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the log-likelihood of data point(s) y."""
        r = jnp.atleast_2d(y) - jnp.atleast_2d(self.mu)

        return (
            -self.N * jnp.log(2 * jnp.pi) / 2
            - jnp.log(self.sigma).sum() / 2
            - ((r / jnp.sqrt(self.sigma)) ** 2).sum(-1) / 2
        )


# %% [markdown]
# We also might have cases where we have a Gaussian which is isotropic, i.e. there is the same independent normal distribution over every dimension. For example you would see this if you have independent measurement noise on some data.


# %%
class IsotropicGaussian(eqx.Module):
    """Multivariate Gaussian distribution with isotropic variance (identity * scalar)."""

    mu: jnp.ndarray  # Shape: (N,)
    sigma: jnp.ndarray  # Shape: () or (1,) - Scalar variance

    @property
    def N(self) -> int:
        """Returns the dimensionality of the distribution."""
        return self.mu.shape[0]

    def sample(self, key: jr.PRNGKey, nsamp: int = 1) -> Tuple[jr.PRNGKey, jnp.ndarray]:
        """Generates random samples from the distribution."""
        key, subkey = jr.split(key)
        ys = self.mu[None, :] + jr.normal(subkey, (nsamp, self.N)) * jnp.sqrt(
            self.sigma
        )
        return key, ys

    def loglik(self, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the log-likelihood of data point(s) y."""
        r = jnp.atleast_2d(y) - jnp.atleast_2d(self.mu)

        return (
            -self.N * jnp.log(2 * jnp.pi) / 2
            - (self.N * jnp.log(self.sigma)) / 2
            - ((r / jnp.sqrt(self.sigma)) ** 2).sum(-1) / 2
        )


# %%
class SE(eqx.Module):
    """Squared Exponential Covariance Function

    Also... Exponentiated Quadratic, RBF, Gaussian...

    Hyperparameters:
        - sf2 (float): signal variance
        - ll (float): squared length scale

    """

    sf2: float
    ll: float

    def k(self, xp: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        """Compute covariance

        Args:
            xp (jnp.ndarray): input x  (D, )
            xq (jnp.ndarray): input x' (D, )

        Returns:
            jnp.ndarray: k(x, x') ( )
        """
        return self.sf2 * jnp.exp(-((xp - xq) ** 2).sum() / 2 / self.ll)


# %%
class SEARD(eqx.Module):
    """Squared Exponential Covariance Function - "Automatic Relavance Determination"

    Also... Exponentiated Quadratic, RBF, Gaussian...

    Hyperparameters:
        - sf2 (float): signal variance
        - ll (jnp.ndarray): (squared) length scale per dimension

    """

    sf2: float
    ll: jnp.ndarray

    def k(self, xp: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        """Compute covariance

        Args:
            xp (jnp.ndarray): input x  (D, )
            xq (jnp.ndarray): input x' (D, )

        Returns:
            jnp.ndarray: k(x, x') ( )
        """
        r = (xp - xq) / jnp.sqrt(self.ll)
        return self.sf2 * jnp.exp(-(r**2).sum() / 2)


# %%
class Mat32(eqx.Module):
    """Matern 3/2 Covariance Function

    General nonlinear covariance function, twice differentiable from \\nu=3/2

    Hyperparameters:
        - sf2 (float): signal variance
        - ll (float): length scale

    """

    sf2: float
    ll: float

    def k(self, xp, xq):
        """Compute covariance

        Args:
            xp (jnp.ndarray): input x  (D, )
            xq (jnp.ndarray): input x' (D, )

        Returns:
            jnp.ndarray: k(x, x') ( )
        """
        r = jnp.sqrt(3) * jnp.sqrt(((xp - xq) ** 2).sum())
        return self.sf2 * (1 + r / self.ll) * jnp.exp(-r / self.ll)


# %%
class Periodic(eqx.Module):
    """Periodic Covariance Function

    Models functions that repeat themselves exactly with a given period.

    Hyperparameters:
        - sf2 (float): signal variance
        - ll (float): length scale (controls the smoothness/wobble)
        - pp (float): period (the distance between repetitions)

    """

    sf2: float
    ll: float
    pp: float

    def k(self, xp, xq):
        """Compute covariance

        Args:
            xp (jnp.ndarray): input x  (D, )
            xq (jnp.ndarray): input x' (D, )

        Returns:
            jnp.ndarray: k(x, x') ( )
        """
        r = jnp.pi * jnp.sqrt(((xp - xq) ** 2).sum()) / self.pp
        return self.sf2 * jnp.exp(-2 * jnp.sin(r) ** 2 / self.ll**2)


# %%
class Linear(eqx.Module):
    """Linear Basis Covariance Function

    All linear in the parameters models can share a covariance function.

    User can provide basis function (N, D) -> (N, F), default identity function

    Hyperparameters:
        - sf2 (float): signal variance (prior variance of the weights)
        - basis (Callable): feature mapping function phi(x) that maps a
                            (D, ) input to a (F, ) feature vector.
                            Defaults to the identity function.

    """

    sf2: float
    basis: Callable[[jnp.ndarray], jnp.ndarray] = lambda xx: xx

    def k(self, xp, xq):
        """Compute covariance

        Args:
            xp (jnp.ndarray): input x  (D, )
            xq (jnp.ndarray): input x' (D, )

        Returns:
            jnp.ndarray: k(x, x') ( )
        """
        return self.sf2 * jnp.inner(self.basis(xp), self.basis(xq))


# %%
class WhiteKernel(eqx.Module):
    sn2: float

    def k(self, xp, xq):
        return jax.lax.cond(jnp.allclose(xp, xq), lambda: self.sn2, lambda: 0.0)


# %%
class SumKernel(eqx.Module):
    k1: Kernel
    k2: Kernel

    def k(self, xp: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        return self.k1.k(xp, xq) + self.k2.k(xp, xq)


class ProductKernel(eqx.Module):
    k1: Kernel
    k2: Kernel

    def k(self, xp: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        return self.k1.k(xp, xq) * self.k2.k(xp, xq)


# Maintain a type to hold our covariance kernels for type hinting
Kernel = (
    SE | SEARD | Mat32 | Periodic | Linear | WhiteKernel | SumKernel | ProductKernel
)


# %%
def covariance(kernel: Kernel) -> Callable:
    """Create Pairwise Covariance Function from Kernel

    Args:
        kernel (Kernel): class of kernel we want to use

    Returns:
        Callabel : function to produce pairwise K_{ab}
    """
    return vmap(vmap(kernel.k, in_axes=(None, 0)), in_axes=(0, None))


def diagonal_covariance(kernel: Kernel) -> Callable:
    """Diagonal covariance of a Gram matrix

    Will return diagonal of square covariance K_{aa}

    Args:
        kernel (Kernel): class of kernel we want to use

    Returns:
        Callabel : function to produce pairwise K_{ii}
    """
    return vmap(lambda xx: kernel.k(xx, xx))


# %%
class ZeroMean(eqx.Module):
    """Zero Mean Function.

    Returns a static mean of zero for all input locations.
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the mean function.

        Args:
            x (jnp.ndarray): Input data array of shape (N, D).

        Returns:
            jnp.ndarray: A vector of zeros of shape (N,).
        """
        return jnp.array([0.0])


class ConstantMean(eqx.Module):
    """Constant Mean Function.

    Returns a static mean for all input locations.
    """

    c: float

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the mean function.

        Args:
            x (jnp.ndarray): Input data array of shape (N, D).

        Returns:
            jnp.ndarray: A vector of zeros of shape (N,).
        """
        return jnp.array([self.c])


class LinearMean(eqx.Module):
    """Linear Basis Mean Function.

    Covariance for all models we can write as:
        k(x,x') = \\sigma_f^2 <\\phi(x), \\phi(x')>

    Including linear, polynomial etc.

    Hyperparameters:
        - w (jnp.ndarray): Weight vector of shape (F,).
        - basis (Callable): Feature mapping function phi(x) that maps
                            an (N, D) input to an (N, F) feature matrix.
    """

    w: jnp.ndarray
    basis: Callable[[jnp.ndarray], jnp.ndarray] = lambda xx: xx

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the mean function.

        Args:
            x (jnp.ndarray): Input data array of shape (N, D).

        Returns:
            jnp.ndarray: Linear mean predictions of shape (N,).
        """
        return jnp.dot(jnp.atleast_1d(self.basis(x)), self.w)


MeanFunction = ZeroMean | ConstantMean | LinearMean


# %%
# GP Functions


class ExactGPState(eqx.Module):
    """State of GP

    We can precompute and store the cholesky factor of \\tilde{K} and the
    "Woodbury" coefficients \\tilde{K} \\alpha = y computed by backsubstitution

    """

    L: jnp.ndarray  # Lower Cholesky of \tilde{K} = K_{ff} (+ \sigma_n^2 I)
    alpha: jnp.ndarray


# %%
class ExactGP(eqx.Module):
    """Exact Gaussian Process Regression Model.

    Work with GP models for

    y = f(x), f(x) ~ GP( m(x), k(x, x') )

    """

    mean: MeanFunction
    kernel: Kernel

    def update_state(self, x: jnp.ndarray, y: jnp.ndarray) -> ExactGPState:
        """Compute GP state from training data.

        Args:
            x (jnp.ndarray): Training inputs of shape (N, D).
            y (jnp.ndarray): Training targets of shape (N, ).

        Returns:
            ExactGPState: State containing important variables.
        """
        N = x.shape[0]
        # Compute the training covariance matrix with a nugget for numerical stability
        ktilde = covariance(self.kernel)(x, x) + NUGGET * jnp.identity(N)
        L = jnp.linalg.cholesky(ktilde)

        # Compute alpha = K^-1 * y via back-substitution: L^T * L * alpha = y - m(x)
        y -= vmap(self.mean)(x).squeeze()
        alpha = solve_triangular(
            L, solve_triangular(L, y, lower=True), lower=True, trans=True
        )

        return ExactGPState(L, alpha)

    def predict(
        self, x: jnp.ndarray, xs: jnp.ndarray, state: ExactGPState
    ) -> MultivariateGaussian:
        """Compute full posterior predictive

        Args:
            x (jnp.ndarray): Training inputs of shape (N, D).
            xs (jnp.ndarray): Test inputs of shape (Ns, D).
            state (ExactGPState): GP state from `update_state`.

        Returns:
            MultivariateGaussian: posterior predictive
        """
        cov = covariance(self.kernel)
        ksx = cov(xs, x)
        kss = cov(xs, xs)

        # m = m(xs) + K_sx * K_xx^-1 * y = m(xs) + K_sx * alpha
        m = vmap(self.mean)(xs).squeeze() + ksx @ state.alpha

        # v = K_ss - K_sx * K_xx^-1 * K_xs
        R = solve_triangular(state.L, ksx.T, lower=True)
        v = kss - R.T @ R

        return MultivariateGaussian(m, v)

    def predict_marginal(
        self, x: jnp.ndarray, xs: jnp.ndarray, state: ExactGPState
    ) -> DiagonalGaussian:
        """Compute marginal predictive

        Often we don't need the cross covariance of the predictive so
        it is more efficient to only compute the diagonal

        Args:
            x (jnp.ndarray): Training inputs of shape (N, D)
            xs (jnp.ndarray): Test inputs of shape (Ns, D)
            state (ExactGPState): GP state from `update_state`

        Returns:
            DiagonalGaussian: marginal posterior distributions
        """
        ksx = covariance(self.kernel)(xs, x)
        kss = diagonal_covariance(self.kernel)(xs)  # Only pull diagonal elements

        m = vmap(self.mean)(xs).squeeze() + ksx @ state.alpha

        R = solve_triangular(state.L, ksx.T, lower=True)
        v = kss - (R**2).sum(0)

        return DiagonalGaussian(m, v)

    def nlml(self, x: jnp.ndarray, y: jnp.ndarray, state: ExactGPState) -> jnp.ndarray:
        """Negative Log Marginal Likelihood

        Args:
            x (jnp.ndarray): Training inputs (N, D).
            y (jnp.ndarray): Training targets (N, ).
            state (ExactGPState): GP state from `update_state`.

        Returns:
            jnp.ndarray: negative log marginal likelihood (scalar)
        """
        N = x.shape[0]
        R = solve_triangular(state.L, y - vmap(self.mean)(x).squeeze(), lower=True)

        # N/2 log(2π) + 0.5 log |Ktilde| + 0.5 *(y' Ktilde^-1 y)
        return (
            N / 2 * jnp.log(2 * jnp.pi)
            + jnp.log(state.L.diagonal()).sum()
            + (R**2).sum() / 2
        )


# %%
def plot_gp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    posterior: Union["MultivariateGaussian", "DiagonalGaussian", "IsotropicGaussian"],
    ax: Optional[plt.Axes] = None,
    nsig: Union[float, int] = 3,
) -> None:
    """
    Plots the true data, the GP mean prediction, and a shaded uncertainty region.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    classname = type(posterior).__name__

    if "Multivariate" in classname:
        std = jnp.sqrt(posterior.sigma.diagonal())
    elif "Diagonal" in classname:
        std = jnp.sqrt(posterior.sigma)
    elif "Isotropic" in classname:
        std = jnp.sqrt(posterior.sigma) * jnp.ones_like(posterior.mu)
    else:
        raise TypeError(f"Unsupported posterior distribution type: {classname}")

    x_plot = jnp.atleast_1d(x).squeeze()
    mean_plot = jnp.atleast_1d(posterior.mu).squeeze()
    std_plot = jnp.atleast_1d(std).squeeze()

    lower_bound = mean_plot - nsig * std_plot
    upper_bound = mean_plot + nsig * std_plot

    ax.plot(x, y, color="black", linestyle="--", label="Observed Data")
    ax.plot(x_plot, mean_plot, color="tab:blue", lw=2, label="GP Mean")

    ax.fill_between(
        x_plot,
        lower_bound,
        upper_bound,
        color="tab:blue",
        alpha=0.2,
        label=f"±{nsig} Std. Dev.",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)


# %%
# GP Hyperparameter Optimisation


def fit(
    data: Tuple[jnp.ndarray, jnp.ndarray], gp: ExactGP, opt, max_iters: int = 200
) -> Tuple[ExactGP, jnp.ndarray]:
    """Fits an Exact Gaussian Process model by optimizing its hyperparameters.

    This function optimises the trainable parameters of an `ExactGP` model by
    minimizing its Negative Log Marginal Likelihood (NLML). It features specific
    handling for kernel stability and parameter constraints:

    1. Parameter Freezing: we don't train the period of periodic kernels
    2. Unconstrained Optimization: we softplus all kernel hyperparameters

    Args:
        data: tuple containing the training data (x, y), shape (N, D), (N, )
        gp: an instance of `ExactGP` to be optimised
        opt: an Optax optimizer instance (e.g., `optax.adam`).
        max_iters: number of optimization steps to perform. Defaults to 200.

    Returns:
        Tuple containing:
            - ExactGP: The optimized Gaussian Process model with updated
            hyperparameters.
            - jnp.ndarray: A record of the loss values (NLML) across all iterations.
    """

    # We never want the period of the periodic kernel to be trained... too unstable.
    trainable, static = eqx.partition(
        gp,
        jax.tree.map_with_path(
            lambda pp, ll: (
                True if pp[-1].name != "pp" and eqx.is_array_like(ll) else False
            ),
            gp,
        ),
    )

    # Apply positive transform on kernel hyperparameters.
    transform_mask = jax.tree_util.tree_map(lambda _: False, trainable)
    transform_mask = eqx.tree_at(
        lambda m: m.kernel,
        transform_mask,
        replace_fn=lambda subtree: jax.tree.map(eqx.is_array_like, subtree),
    )

    # Softplus and inverse for our positive only hyperparameters
    def _param_mapping(tree):
        return jax.tree.map(
            lambda x, mask: jnp.log(1 + jnp.exp(x)) if mask else x, tree, transform_mask
        )

    def _inv_param_mapping(tree):
        return jax.tree.map(
            lambda x, mask: jnp.log(jnp.exp(x) - 1) if mask else x, tree, transform_mask
        )

    # Move parameters to training space
    trainable = _inv_param_mapping(trainable)

    # Training objective: remap parameters, update GP, compute marginal likelihood
    def _cost(trainable: ExactGP, static: ExactGP):
        trainable = _param_mapping(trainable)
        model = eqx.combine(trainable, static)
        state = model.update_state(*data)
        cost = model.nlml(*data, state)
        return cost

    # Actual value and gradient calculation only on trainable part
    _cost_frozen = jax.value_and_grad(jax.jit(partial(lambda tt: _cost(tt, static))))

    # Optimisation step
    def _step(carry, xs):
        train, opt_state = carry
        v, g = _cost_frozen(train)
        updates, opt_state = opt.update(g, opt_state)
        train = optax.apply_updates(train, updates)
        return (train, opt_state), v

    # Perform the optimisation for max_iters epochs
    opt_state = opt.init(trainable)
    (trainable, opt_state), cost_log = jax.lax.scan(
        _step, (trainable, opt_state), None, length=max_iters
    )

    # Rebuild the model before returning
    return (eqx.combine(_param_mapping(trainable), static), cost_log)


# %%
def plot_2d_gp(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], Any],
    grid_res: int = 50,
    padding: float = 0.5,
):
    """
    Plots a 3D surface of a 2-input GP model's mean and uncertainty alongside training data.

    Parameters:
    -----------
    X : np.ndarray
        Training inputs, shape (N, 2)
    y : np.ndarray
        Training targets, shape (N,)
    predict_fn : Callable[[np.ndarray], DiagonalGaussian]
        A function that accepts a (M, 2) test input array and returns
        the DiagonalGaussian object containing .mu and .sigma
    grid_res : int, optional
        Resolution of the prediction grid mesh (grid_res x grid_res), default is 50.
    padding : float, optional
        How far past the min/max training data points to extend the plot, default is 0.5.
    """
    # 1. Setup the Grid for Predictions
    x_bounds = [(X[:, i].min() - padding, X[:, i].max()) for i in range(X.shape[1])]

    x_grid = np.meshgrid(
        *[
            np.linspace(x_bounds[i][0], x_bounds[i][1], grid_res)
            for i in range(X.shape[1])
        ],
    )

    # Flatten grid for the prediction function
    X_test = np.vstack([xx.ravel() for xx in x_grid]).T

    # 2. Get GP Predictions via the callback
    prediction = predict_fn(X_test)

    # Reshape outputs back into 2D grid dimensions
    # Note: jnp arrays from Equinox/JAX convert cleanly to numpy via np.array
    mu_grid = np.array(prediction.mu).reshape(x_grid[0].shape)
    std_grid = np.sqrt(np.array(prediction.sigma)).reshape(x_grid[0].shape)

    # 3. 3D Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot for training data
    ax.scatter(
        X[:, 0],
        X[:, 1],
        y,
        color="red",
        s=40,
        zorder=10,
        label="Training Data",
        edgecolor="k",
    )

    # Mean Prediction Surface
    surf = ax.plot_surface(
        x_grid[0],
        x_grid[1],
        mu_grid,
        cmap="viridis",
        alpha=0.6,
        linewidth=0,
        antialiased=True,
        label="Predicted Mean ($\\mu$)",
    )
    # Workaround for matplotlib surface legend bug
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # Uncertainty Bounds (95% Confidence Interval)
    ax.plot_surface(
        x_grid[0],
        x_grid[1],
        mu_grid + 2 * std_grid,
        color="gray",
        alpha=0.15,
        linewidth=0,
        antialiased=True,
    )
    ax.plot_surface(
        x_grid[0],
        x_grid[1],
        mu_grid - 2 * std_grid,
        color="gray",
        alpha=0.15,
        linewidth=0,
        antialiased=True,
    )

    # Labels and Styling
    ax.set_xlabel("Input Dimension 1 ($X_1$)")
    ax.set_ylabel("Input Dimension 2 ($X_2$)")
    ax.set_zlabel("Output ($y$)")
    ax.set_title("2D Input Gaussian Process Regression")
    ax.legend()

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.show()


# %%
class Normaliser(eqx.Module):
    @staticmethod
    def self_normalise(
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        m = x.mean(0, keepdims=True)
        s = x.std(0, keepdims=True)

        return (x - m) / s, (m, s)

    @staticmethod
    def normalise(
        x: jnp.ndarray, scales: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        m, s = scales
        return (x - m) / s

    @staticmethod
    def denormalise(
        x: jnp.ndarray, scales: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        m, s = scales
        return x * s + m
