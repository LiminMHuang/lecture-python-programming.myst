---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

<a id='sp'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>

+++

# SciPy


<a id='index-1'></a>
In addition to what is in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:hide-output: false

!pip install --upgrade quantecon
```

We use the following imports

```{code-cell} ipython3
:hide-output: false

import numpy as np
import quantecon as qe
```

## Overview

[SciPy](https://scipy.org/) builds on top of NumPy to provide common tools for scientific programming such as

- [linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [numerical integration](https://docs.scipy.org/doc/scipy/reference/integrate.html)
- [interpolation](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
- [optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [distributions and random number generation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- etc.


Like NumPy, SciPy is stable, mature and widely used.

Many SciPy routines are thin wrappers around industry-standard Fortran libraries such as [LAPACK](https://en.wikipedia.org/wiki/LAPACK), [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), etc.

It is not really necessary to “learn” SciPy as a whole.

A more common approach is to get some idea of what is in the library and then look up [documentation](https://docs.scipy.org/doc/scipy/reference/index.html) as required.

In this lecture, we aim only to highlight some useful parts of the package.

+++

## SciPy versus NumPy

SciPy is a package that contains various tools that are built on top of NumPy, using its array data type and related functionality.

>**Note**
>
>In older versions of SciPy (`scipy < 0.15.1`), importing the package would also import NumPy symbols into the global namespace, as can be seen from this excerpt the SciPy initialisation file:

```{code-cell} ipython3
:hide-output: false

from numpy import *
from numpy.random import rand, randn
from numpy.fft import fft, ifft
from numpy.lib.scimath import *
```

However, it is better practice to use NumPy functionality explicitly.

```{code-cell} ipython3
:hide-output: false

import numpy as np

a = np.identity(3)
```

More recent versions of SciPy (1.15+) no longer automatically import NumPy symbols.

What is useful in SciPy is the functionality in its sub-packages

- `scipy.optimize`, `scipy.integrate`, `scipy.stats`, etc.


Let us explore some of the major sub-packages.

+++

## Statistics


<a id='index-4'></a>
The `scipy.stats` subpackage supplies

- numerous random variable objects (densities, cumulative distributions, random sampling, etc.)
- some estimation procedures
- some statistical tests

+++

### Random Variables and Distributions

Recall that `numpy.random` provides functions for generating random variables

```{code-cell} ipython3
:hide-output: false

np.random.beta(5, 5, size=3)
```

This generates a draw from the distribution with the density function below when `a, b = 5, 5`


<a id='equation-betadist2'></a>
$$
f(x; a, b) = \frac{x^{(a - 1)} (1 - x)^{(b - 1)}}
    {\int_0^1 u^{(a - 1)} (1 - u)^{(b - 1)} du}
    \qquad (0 \leq x \leq 1) \tag{13.1}
$$

Sometimes we need access to the density itself, or the cdf, the quantiles, etc.

For this, we can use `scipy.stats`, which provides all of this functionality as well as random number generation in a single consistent interface.

Here is an example of usage

```{code-cell} ipython3
:hide-output: false

from scipy.stats import beta
import matplotlib.pyplot as plt

q = beta(5, 5)      # Beta(a, b), with a = b = 5
obs = q.rvs(2000)   # 2000 observations
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, q.pdf(grid), 'k-', linewidth=2)
plt.show()
```

The object `q` that represents the distribution has additional useful methods, including

```{code-cell} ipython3
:hide-output: false

q.cdf(0.4)      # Cumulative distribution function
```

```{code-cell} ipython3
:hide-output: false

q.ppf(0.8)      # Quantile (inverse cdf) function
```

```{code-cell} ipython3
:hide-output: false

q.mean()
```

The general syntax for creating these objects that represent distributions (of type `rv_frozen`) is

> `name = scipy.stats.distribution_name(shape_parameters, loc=c, scale=d)`


Here `distribution_name` is one of the distribution names in [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).

The `loc` and `scale` parameters transform the original random variable
$ X $ into $ Y = c + d X $.

+++

### Alternative Syntax

There is an alternative way of calling the methods described above.

For example, the code that generates the figure above can be replaced by

```{code-cell} ipython3
:hide-output: false

obs = beta.rvs(5, 5, size=2000)
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, beta.pdf(grid, 5, 5), 'k-', linewidth=2)
plt.show()
```

### Other Goodies in scipy.stats

There are a variety of statistical functions in `scipy.stats`.

For example, `scipy.stats.linregress` implements simple linear regression

```{code-cell} ipython3
:hide-output: false

from scipy.stats import linregress

x = np.random.randn(200)
y = 2 * x + 0.1 * np.random.randn(200)
gradient, intercept, r_value, p_value, std_err = linregress(x, y)
gradient, intercept
```

To see the full list, consult the [documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-functions-scipy-stats).

+++

## Roots and Fixed Points

A **root** or **zero** of a real function $ f $ on $ [a,b] $ is an $ x \in [a, b] $ such that $ f(x)=0 $.

For example, if we plot the function


<a id='equation-root-f'></a>
$$
f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1 \tag{13.2}
$$

with $ x \in [0,1] $ we get

```{code-cell} ipython3
:hide-output: false

f = lambda x: np.sin(4 * (x - 1/4)) + x + x**20 - 1
# lambda is a keyword used to create anonymous (unnamed) functions — quick, one-line functions that need not formally define with 'def'.
x = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f(x)$')
ax.axhline(ls='--', c='k') # This draws a black dashed horizontal line, usually at y = 0.
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.legend(fontsize=12)
plt.show()
```

The unique root is approximately 0.408.

Let us consider some numerical techniques for finding roots.

+++

### Bisection


<a id='index-6'></a>
One of the most common algorithms for numerical root-finding is *bisection*.

To understand the idea, recall the well-known game where

- Player A thinks of a secret number between 1 and 100
- Player B asks if it is less than 50
  - If yes, B asks if it is less than 25
  - If no, B asks if it is less than 75

And so on.

This is bisection.

Here is a simplistic implementation of the algorithm in Python.

It works for all sufficiently well behaved increasing continuous functions with $ f(a) < 0 < f(b) $


<a id='bisect-func'></a>

```{code-cell} ipython3
:hide-output: false

def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b

    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        if f(middle) > 0:   # root is between lower and middle
            lower, upper = lower, middle
        else:               # root is between middle and upper
            lower, upper = middle, upper

    return 0.5 * (upper + lower)
```

Let us test it using the function $ f $ defined in [(13.2)](#equation-root-f)

```{code-cell} ipython3
:hide-output: false

bisect(f, 0, 1)
```

Not surprisingly, SciPy provides its own bisection function.

Let us test it using the same function $ f $ defined in [(13.2)](#equation-root-f)

```{code-cell} ipython3
:hide-output: false

from scipy.optimize import bisect

bisect(f, 0, 1)
```

### The Newton-Raphson Method


<a id='index-8'></a>
Another very common root-finding algorithm is the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method).

In SciPy this algorithm is implemented by `scipy.optimize.newton`.

Unlike bisection, the Newton-Raphson method uses local slope information in an attempt to increase the speed of convergence.

Let us investigate this using the same function $ f $ defined above.

With a suitable initial condition for the search we get convergence:

```{code-cell} ipython3
:hide-output: false

from scipy.optimize import newton

newton(f, 0.2)   # Start the search at initial condition x = 0.2
```

But other initial conditions lead to failure of convergence:

```{code-cell} ipython3
:hide-output: false

newton(f, 0.7)   # Start the search at x = 0.7 instead
```

### Hybrid Methods

A general principle of numerical methods is as follows:

- If you have specific knowledge about a given problem, you might be able to exploit it to generate efficiency.
- If not, then the choice of algorithm involves a trade-off between speed and robustness.


In practice, most default algorithms for root-finding, optimisation and fixed points use *hybrid* methods.

These methods typically combine a fast method with a robust method in the following manner:

1. Attempt to use a fast method
1. Check diagnostics
1. If diagnostics are bad, then switch to a more robust algorithm


In `scipy.optimize`, the function `brentq` is such a hybrid method and a good default

```{code-cell} ipython3
:hide-output: false

from scipy.optimize import brentq

brentq(f, 0, 1)
```

Here the correct solution is found and the speed is better than bisection:

```{code-cell} ipython3
:hide-output: false

with qe.Timer(unit="milliseconds"):
    brentq(f, 0, 1)
```

```{code-cell} ipython3
:hide-output: false

with qe.Timer(unit="milliseconds"):
    bisect(f, 0, 1)
```

### Multivariate Root-Finding


<a id='index-9'></a>
Use `scipy.optimize.fsolve`, a wrapper for a hybrid method in MINPACK.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html) for details.

+++

### Fixed Points

A **fixed point** of a real function $ f $ on $ [a,b] $ is an $ x \in [a, b] $ such that $ f(x)=x $.


<a id='index-10'></a>
SciPy has a function for finding (scalar) fixed points too

```{code-cell} ipython3
:hide-output: false

from scipy.optimize import fixed_point

fixed_point(lambda x: x**2, 10.0)  # 10.0 is an initial guess
```

If you do not get good results, you can always switch back to the `brentq` root finder, since the fixed point of a function $ f $ is the root of $ g(x) := x - f(x) $.

+++

## Optimization


<a id='index-12'></a>
Most numerical packages provide only functions for *minimization*.

Maximization can be performed by recalling that the maximizer of a function $ f $ on domain $ D $ is
the minimizer of $ -f $ on $ D $.

Minimization is closely related to root-finding: For smooth functions, interior optima correspond to roots of the first derivative.

The speed/robustness trade-off described above is present with numerical optimisation too.

Unless you have some prior information you can exploit, it is usually best to use hybrid methods.

For constrained, univariate (i.e., scalar) minimisation, a good hybrid option is `fminbound`

```{code-cell} ipython3
:hide-output: false

from scipy.optimize import fminbound

fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]
```

### Multivariate Optimisation


<a id='index-13'></a>
Multivariate local optimisers include `minimize`, `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, and `fmin_ncg`.

Constrained multivariate local optimisers include `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html) for details.

+++

## Integration


<a id='index-15'></a>
Most numerical integration methods work by computing the integral of an approximating polynomial.

The resulting error depends on how well the polynomial fits the integrand, which in turn depends on how “regular” the integrand is.

In SciPy, the relevant module for numerical integration is `scipy.integrate`.

A good default for univariate integration is `quad`

```{code-cell} ipython3
:hide-output: false

from scipy.integrate import quad

integral, error = quad(lambda x: x**2, 0, 1)
integral
```

In fact, `quad` is an interface to a very standard numerical integration routine in the Fortran library QUADPACK.

It uses [Clenshaw-Curtis quadrature](https://en.wikipedia.org/wiki/Clenshaw-Curtis_quadrature),  based on expansion in terms of Chebychev polynomials.

There are other options for univariate integration — a useful one is `fixed_quad`, which is fast and hence works well inside `for` loops.

There are also functions for multivariate integration.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/integrate.html) for more details.

+++

## Linear Algebra


<a id='index-17'></a>
We saw that NumPy provides a module for linear algebra called `linalg`.

SciPy also provides a module for linear algebra with the same name.

The latter is not an exact superset of the former, but overall it has more functionality.

We leave you to investigate the [set of available routines](https://docs.scipy.org/doc/scipy/reference/linalg.html).

+++

## Exercises

The first few exercises concern pricing a European call option under the assumption of risk neutrality. The price satisfies

$$
P = \beta^n \mathbb E \max\{ S_n - K, 0 \}
$$

where

1. $ \beta $ is a discount factor,
1. $ n $ is the expiry date,
1. $ K $ is the strike price and
1. $ \{S_t\} $ is the price of the underlying asset at each time $ t $.


For example, if the call option is to buy stock in Amazon at strike price $ K $, the owner has the right (but not the obligation) to buy 1 share in Amazon at price $ K $ after $ n $ days.

The payoff is therefore $ \max\{S_n - K, 0\} $

The price is the expectation of the payoff, discounted to current value.

+++

## Exercise 13.1

Suppose that $ S_n $ has the [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) distribution with parameters $ \mu $ and $ \sigma $.  Let $ f $ denote the density of this distribution. Then

$$
P = \beta^n \int_0^\infty \max\{x - K, 0\} f(x) dx
$$

Plot the function

$$
g(x) = \beta^n  \max\{x - K, 0\} f(x)
$$

over the interval $ [0, 400] $ when `μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40`.

From `scipy.stats` you can import `lognorm` and then use `lognorm.pdf(x, σ, scale=np.exp(μ))` to get the density $ f $.

```{code-cell} ipython3
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 400, 1200)
μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40
# Using list comprehension
y = [β**n * max((x_val - K), 0) * lognorm.pdf(x_val, σ, scale=np.exp(μ)) for x_val in x]

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

An improved version

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

x = np.linspace(0, 400, 1200)
μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40

# Using NumPy vectorisation
y = β**n * np.maximum(x - K, 0) * lognorm.pdf(x, σ, scale=np.exp(μ))

fig, ax = plt.subplots()
# label = '$ g(x) = β**n max{x - K, 0}f(x)$'
ax.plot(x, y, linewidth=2, alpha = 0.6)
ax.set_xlabel('$x$')
ax.set_ylabel('$g(x)$')
# ax.legend()
title = r'$ g(x) = \beta^n \max\{x - K, 0\} f(x)$'
ax.set_title(rf'Plot of the integrand {title}')

plt.show()
```

From ChatGPT

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Parameters
μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40

# x grid
x = np.linspace(0, 400, 1200)

# Function definition
y = β**n * np.maximum(x - K, 0) * lognorm.pdf(x, σ, scale=np.exp(μ))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Legend text: three lines (formula, f(x) definition, parameters)
label = (
    r'$g(x) = \beta^n \max\{x - K, 0\} f(x)$' '\n'
    r'$f(x) = \dfrac{1}{x\sigma\sqrt{2\pi}}'
    r'\exp\!\left(-\dfrac{(\ln x - \mu)^2}{2\sigma^2}\right)$' '\n'
    r'$\mu=4,\ \sigma=0.25,\ \beta=0.99,\ n=10,\ K=40$'
)

# Plot line
ax.plot(x, y, linewidth=2, alpha=0.7, color='navy', label=label)

# Labels and title
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$g(x)$', fontsize=12)
ax.set_title(r'Plot of the integrand $g(x)$', fontsize=14, pad=10)

# Legend styling
ax.legend(
    loc='upper right',
    fontsize=9.5,
    frameon=True,
    framealpha=0.9,
    fancybox=True,
    borderpad=1.1,
    handlelength=2.5
)

# Minor layout and visual polish
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Solution to[ Exercise 13.1](https://python-programming.quantecon.org/#sp_ex01)

Here is one possible solution

```{code-cell} ipython3
:hide-output: false

from scipy.integrate import quad
from scipy.stats import lognorm

μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40

def g(x):
    return β**n * np.maximum(x - K, 0) * lognorm.pdf(x, σ, scale=np.exp(μ))

x_grid = np.linspace(0, 400, 1000)
y_grid = g(x_grid) 

fig, ax = plt.subplots()
ax.plot(x_grid, y_grid, label="$g$")
ax.legend()
plt.show()
```

## Exercise 13.2

In order to get the option price, compute the integral of this function numerically using `quad` from `scipy.integrate`.

+++

## Solution to[ Exercise 13.2](https://python-programming.quantecon.org/#sp_ex02)

```{code-cell} ipython3
:hide-output: false

price, error = quad(g, 0, np.inf)  # integrate from 0 to infinity
print(f"Option price ≈ {price:.4f}, estimated error ≈ {error:.2e}")
```

## Exercise 13.3

Try to get a similar result using Monte Carlo to compute the expectation term in the option price, rather than `quad`.

In particular, use the fact that if $ S_n^1, \ldots, S_n^M $ are independent draws from the lognormal distribution specified above, then, by the law of large numbers,

$$
\mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$

Set `M = 10_000_000`

```{code-cell} ipython3
from scipy.stats import lognorm

M = 10_000_000
draws = lognorm.rvs(σ, scale=np.exp(μ), size=M)
P = β**n * np.mean(np.maximum(draws - K, 0))
print(f'The price from Monte Carlo estimation is {P: .3f}.')
```

## Solution to[ Exercise 13.3](https://python-programming.quantecon.org/#sp_ex03)

Here is one solution:

```{code-cell} ipython3
:hide-output: false

M = 10_000_000
S = np.exp(μ + σ * np.random.randn(M))
return_draws = np.maximum(S - K, 0)
P = β**n * np.mean(return_draws) 
print(f"The Monte Carlo option price is {P:3f}")
```

## Exercise 13.4

In [this lecture](https://python-programming.quantecon.org/functions.html#functions), we discussed the concept of [recursive function calls](https://python-programming.quantecon.org/functions.html#recursive-functions).

Try to write a recursive implementation of the homemade bisection function [described above](#bisect-func).

Test it on the function [(13.2)](#equation-root-f).

+++

## Solution to[ Exercise 13.4](https://python-programming.quantecon.org/#sp_ex1)

Here is a reasonable solution:

```{code-cell} ipython3
:hide-output: false

def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root-finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b
    if upper - lower < tol:
        return 0.5 * (upper + lower)
    else:
        middle = 0.5 * (upper + lower)
        print(f'Current mid point = {middle}')
        if f(middle) > 0:   # Implies root is between lower and middle
            return bisect(f, lower, middle) # This is recursive because it calls a function within the function itself.
        else:               # Implies root is between middle and upper
            return bisect(f, middle, upper)
```

We can test it as follows

```{code-cell} ipython3
:hide-output: false

f = lambda x: np.sin(4 * (x - 0.25)) + x + x**20 - 1
bisect(f, 0, 1)
```
