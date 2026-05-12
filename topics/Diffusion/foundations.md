# Foundations of probability — Xinyuan

---
## What is ELBO? Why do we use it?
**T:** Basic

**A:** 
The **ELBO** is the **Evidence Lower BOund** on the marginal log-likelihood $\log p_\theta(x)$, typically used in latent-variable models where
$$
p_\theta(x)=\int p_\theta(x,z)\,dz
$$
is hard to compute exactly.

For any auxiliary distribution $q_\phi(z\mid x)$,
$$\log p_\theta(x)\;\ge\;\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].$$

This lower bound is the ELBO:
$$\mathcal{L}(\theta,\phi;x)=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x,z)]-\mathbb{E}_{q_\phi(z\mid x)}[\log q_\phi(z\mid x)].$$

A more interpretable form is

$$
\mathcal{L}(\theta,\phi;x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right).
$$

So ELBO has two terms:

- a **data-fit / reconstruction term**: make $z$ explain $x$,
- a **regularization term**: keep the approximate posterior close to the prior.

Why use it? Because exact Bayesian learning or exact MLE in latent-variable models often requires the posterior
$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)},
$$
and both the denominator $p_\theta(x)$ and the posterior are intractable. The ELBO converts the problem into an optimization problem over a tractable surrogate.

Crucially,
$$
\log p_\theta(x)=\mathcal{L}(\theta,\phi;x)+\mathrm{KL}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right).
$$
So maximizing the ELBO does two things at once:

1. it increases a lower bound on the true log-evidence,
2. it makes $q_\phi(z\mid x)$ closer to the true posterior.

This is why ELBO is central in **variational inference**, **VAEs**, and many Bayesian latent-variable models.

**In short**

- ELBO = tractable lower bound on $\log p_\theta(x)$
- used when exact marginalization/posterior inference is intractable
- maximization = approximate inference + approximate learning

---

## Why can we not maximize MLE directly?
**T:** Deep

**A:** 
The careful answer is: **sometimes we can**. In simple fully observed models, direct MLE is straightforward.

We **cannot maximize MLE directly** when the likelihood itself or its gradient is intractable. This happens most often in:

**(a) Latent-variable models**

If
$$
p_\theta(x)=\int p_\theta(x,z)\,dz,
$$
then the marginalization over $z$ may be analytically impossible or computationally prohibitive.

Even if we write the likelihood,

$$
\ell(\theta)=\sum_i \log p_\theta(x_i),
$$

its gradient often involves the posterior:

$$
\nabla_\theta \log p_\theta(x)
=\mathbb{E}_{p_\theta(z\mid x)}\left[\nabla_\theta \log p_\theta(x,z)\right],
$$
and $p_\theta(z\mid x)$ is precisely what we do not know how to compute.

**(b) Models with an intractable partition function**

In EBMs,

$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta},\qquad
Z_\theta=\int e^{-E_\theta(x)}dx.
$$

MLE requires $\log Z_\theta$, whose gradient is

$$
\nabla_\theta \log Z_\theta
=\mathbb{E}_{p_\theta(x)}[-\nabla_\theta E_\theta(x)].
$$
That expectation is often expensive because it requires sampling from the model itself.

**(c) Nonconvex high-dimensional optimization**

Even when the likelihood is explicit, direct maximization may still be numerically difficult due to local optima, saddle points, poor conditioning, or huge datasets.

So the real issue is not “MLE is impossible” in general. The issue is that **direct maximization of the exact likelihood can be computationally intractable**, and we therefore use approximations such as:

- **EM**,
- **variational inference / ELBO**,
- **MCMC-based approximations**,
- **contrastive divergence**, **score matching**, **noise-contrastive estimation**, etc.

**In short:**

- direct MLE is fine in simple observed models
- fails in practice when likelihood or gradient is intractable
- main reasons: latent variables, partition functions, hard optimization
- ELBO/EM/MCMC replace exact MLE with tractable surrogates

---

## How to prove / derive the ELBO?
**T:** Deep

**A:** 
There are two standard derivations.

**Derivation via Jensen**

Start with

$$
\log p_\theta(x)=\log \int p_\theta(x,z)\,dz.
$$

Insert $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=\log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
=\log \mathbb{E}_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

By Jensen,

$$
\log \mathbb{E}_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\ge
\mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

So

$$
\log p_\theta(x)\ge
\mathbb{E}_{q_\phi}[\log p_\theta(x,z)-\log q_\phi(z\mid x)].
$$

**Derivation via KL**

Start from

$$
\mathrm{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))
=\mathbb{E}_{q_\phi}\left[\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\right].
$$

Use

$$
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.
$$

Then rearrange to get

$$
\log p_\theta(x)=\text{ELBO}+\mathrm{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x)).
$$

**What to emphasize:**

The KL derivation is the best one to say in an interview, because it makes the approximation gap explicit.

**Likely follow-up:**

**Which derivation is more insightful?**
 Jensen explains the lower bound; the KL derivation explains the optimization meaning.

**In short:**

- introduce $q(z\mid x)$
- Jensen gives lower bound
- KL derivation gives exact decomposition
- bound is tight when $q=p(z\mid x)$



---

## What is EBM? Why do we need this when we have MLE / KL divergence?
**T:** Basic

**A:** 
An **Energy-Based Model** defines
$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta}.
$$
The model assigns lower energy to more plausible states.

MLE and KL divergence are **training principles**. An EBM is a **model class**. We need EBMs because in many problems it is much easier to define a scalar compatibility score than a tractable normalized likelihood.

**What to emphasize:**

EBMs are useful when:

- normalization is hard,
- dependencies are complex,
- structured outputs matter,
- relative plausibility matters more than explicit density factorization.

In structured prediction, one often writes
$$
\hat y = \arg\min_y E_\theta(x,y).
$$

**Likely follow-up:**

**If $Z_\theta$ is intractable, how do you train them?**
 Approximate MLE, contrastive divergence, score matching, noise-contrastive estimation, or score-based objectives.

**How is this related to KL?**
 MLE still corresponds to minimizing $\mathrm{KL}(p_{\text{data}}\|p_\theta)$, but computing it is difficult because $p_\theta$ is hard to normalize.

**In short:**

- EBM = $p(x)\propto e^{-E(x)}$
- energy low = plausible
- EBM is a model family, not an objective
- useful when normalized likelihood is hard to specify

---

## What's the difference between min, average, and expectation?
**T:** Basic

**A:** 
These are three different operations.

- **Minimum**:

$$
\min_\theta f(\theta)
$$

is an optimization operator.

- **Average**:

$$
\frac1n\sum_{i=1}^n f(x_i)
$$

is the empirical mean over a sample.

- **Expectation**:

$$
\mathbb{E}[f(X)] = \int f(x)p(x)\,dx
$$

is the population mean under a probability distribution.

**What to emphasize:**

Expectation is theoretical; average is sample-based; minimum is optimization.

A central statistical distinction is:
$$
\min_\theta \frac1n \sum_{i=1}^n L(\theta,x_i)
$$
versus
$$
\min_\theta \mathbb{E}[L(\theta,X)].
$$
The first is empirical risk minimization; the second is population risk minimization.

**Likely follow-up:**

**Do min and expectation commute?**
 Usually no:
$$
\min_\theta \mathbb{E}[L(\theta,X)] \neq \mathbb{E}[\min_\theta L(\theta,X)].
$$

**In short:**

- minimum = optimization
- average = sample mean
- expectation = distribution mean
- average estimates expectation
- minimum and expectation usually do not commute
---
### What's importance sampling
**T:** Basic

**:A** 
Importance sampling estimates an expectation under a target distribution $p$ by sampling from a proposal $q$.

We want

$$
\mathbb{E}_p[f(X)] = \int f(x)p(x)\,dx.
$$

If we can sample from $q$, then

$$
\mathbb{E}_p[f(X)]
=\mathbb{E}_q\left[f(X)\frac{p(X)}{q(X)}\right].
$$

So with $x^{(i)}\sim q$,
$$
\hat I = \frac1N\sum_{i=1}^N f(x^{(i)})\frac{p(x^{(i)})}{q(x^{(i)})}.
$$

**What to emphasize:**

The weight
$$
w(x)=\frac{p(x)}{q(x)}
$$
corrects for the mismatch between proposal and target.

The main issue is variance: if $q$ misses important regions, weights explode and the estimator becomes unstable.

**Likely follow-up:**

**What is the support condition?**
 If $p(x)>0$, then we need $q(x)>0$.

**What if only an unnormalized target is known?**
 Use self-normalized importance sampling.

**In short:**

- sample from easy $q$, estimate under hard $p$
- correct with weights $p/q$
- requires support coverage
- works well only with a good proposal
---

## Get 6/7 probability out of a 6-side dice
**T:** Hands-on

**A:** 
Assuming the question means: construct an event with exact probability $6/7$ using repeated fair die rolls.

Roll the die twice. That gives $36$ equally likely ordered outcomes. Reject one outcome, say $(6,6)$, and reroll whenever it happens. Then the remaining $35$ accepted outcomes are equally likely.

Now label $30$ of those $35$ outcomes as “success” and the other $5$ as “failure”. Then
$$
\mathbb{P}(\text{success})=\frac{30}{35}=\frac67.
$$

**What to emphasize:**

This is an exact construction using rejection sampling.

**Likely follow-up:**

**Why not do it in one roll?**
 Because a single fair die only gives probabilities that are multiples of $1/6$, and $6/7$ is not one of them.

**In short:**

- one roll cannot realize $6/7$
- two rolls give $36$ outcomes
- reject one outcome to get $35$ equiprobable states
- choose 30 successes: $30/35=6/7$

---

## What is Ornstein-Uhlenbeck process? What is Brownian motion? What’s the difference?
**T:** Deep

**A:** 
### Brownian motion

A standard Brownian motion $(W_t)$ satisfies:

- $W_0=0$,
- continuous paths,
- independent increments,
- $W_t-W_s\sim \mathcal N(0,t-s)$.

It solves
$$
dX_t=dW_t.
$$

### Ornstein-Uhlenbeck process

The OU process solves
$$
dX_t=-\theta(X_t-\mu)\,dt+\sigma\,dW_t,
\qquad \theta>0.
$$
It is a Gaussian Markov process with mean reversion toward $\mu$.

**What to emphasize:**

Brownian motion is pure diffusion. OU is diffusion plus restoring drift.

For Brownian motion:

- mean $0$,
- variance grows like $t$,
- nonstationary.

For OU:

- mean returns to $\mu$,
- stationary distribution exists,
- temporal correlations decay exponentially.

**Likely follow-up:**

**Is OU Markov?**
 Yes.

**Is Brownian motion stationary?**
 No; increments are stationary, but the process itself is not.

**In short:**

- Brownian: no restoring force, variance grows forever
- OU: mean-reverting diffusion
- Brownian has independent increments
- OU has correlated increments and stationary limit
---

### 4 cases and 8 books, at least 1 in each case: how many ways?
**T:** Hands-on

**:A** 
Assume the 8 books are distinct and the 4 cases are distinct.

We count surjections from 8 books to 4 cases. By inclusion-exclusion:
$$
4^8-\binom41 3^8+\binom42 2^8-\binom43 1^8.
$$
Numerically,
$$
65536-4\cdot6561+6\cdot256-4=40824.
$$

**What to emphasize:**

This is a standard onto-function counting problem.

Equivalent form:
$$
4!S(8,4),
$$
where $S(8,4)$ is a Stirling number of the second kind.

**Likely follow-up:**

**What if books were identical?**
 Then it becomes an integer partition/composition problem, not a surjection count.

**In short:**

- distinct books, distinct cases
- onto maps from 8 items to 4 bins
- inclusion-exclusion
- answer = $40824$
---

## What is Markov property? How to model a non-Markov process as a Markov process?
**T:** Basic

**A:** 

A process is **Markov** if the future depends on the past only through the present:

$$
\mathbb{P}(X_{t+1}\in A\mid X_t,X_{t-1},\dots)
=\mathbb{P}(X_{t+1}\in A\mid X_t).
$$

**What to emphasize:**

The present state is a sufficient statistic of the past for predicting the future.

To convert a non-Markov process into a Markov one, augment the state with enough memory.

For a $k$-th order chain, define
$$
Y_t=(X_t,\dots,X_{t-k+1}),
$$
then $Y_t$ is first-order Markov.

For AR($p$):
$$
X_t=\sum_{i=1}^p a_i X_{t-i}+\varepsilon_t,
$$
define
$$
Y_t=(X_t,\dots,X_{t-p+1}).
$$

**Likely follow-up:**

**Can every non-Markov process be made Markov?**
 Yes in principle by taking the whole past as the state, but that may be infinite-dimensional.

**In short:**

- Markov = future independent of past given present
- present summarizes predictive information
- non-Markov can be Markovized by state augmentation
- finite augmentation works when memory is finite

---

## Define EBM and where they are used
**T:** Basic

**A:** 
An EBM defines an energy $E_\theta(x)$ or $E_\theta(x,y)$ and treats lower energy as higher compatibility.

Unconditional form:
$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta}.
$$
Conditional form:
$$
p_\theta(y\mid x)\propto e^{-E_\theta(x,y)}.
$$

**What to emphasize:**

EBMs are useful wherever scoring compatibility is easier than constructing a normalized explicit likelihood.

Typical applications:

- generative modeling,
- structured prediction,
- computer vision,
- inverse RL,
- anomaly detection,
- score-based modeling.

**Likely follow-up:**

**What is the main difficulty?**
 Partition function estimation and sampling.

**Why are they powerful?**
 Because they can represent complex dependencies without restrictive factorization assumptions.

**In short:**

- EBM = learn an energy landscape
- low energy = data-like / compatible
- used in vision, structure prediction, RL, anomaly detection
- expressive but hard to normalize and sample from
---
