# Probabilistic Q-Learning

This is my attempt to figure out how Q-learning should be trained from first-principles

# Some background

Off-policy reinforcement learning revolves around the **Bellman equation**:

$$
Q^*(s,a) = r(s,a) + \gamma \max_{a'} Q^*(s',a')
$$

Where:

- $Q^*(s,a)$ is the optimal $Q$-value for taking action $a$ in state $s$. It represents the maximum total discounted reward the agent can expect from this point forward.
- $r(s,a)$ is the immediate reward received after taking action $a$ from state $s$.
- $\gamma \in (0,1]$ is the discount factor.
- $\max_{a'}Q^*(s',a')$ is the maximum $Q$-value for the next state $s'$ across all possible actions $a'$. This term links the current $Q$-value to the optimal future value, assuming the agent acts optimally.


## The current Loss Function

Training is typically **bootstrapped**. A common approach is to minimize the $L_2$ loss:

$$
\mathbb{E}_{\{s,a,r,s'\}\sim D}\left[ Q_\theta(s,a) - \left( r + \gamma \max_{a'} Q_{\bar\theta}(s',a') \right) \right]^2
$$

where $D$ is the replay buffer, $\theta$ are the learnable parameters, and $\bar\theta$ is a delayed copy of the network used for stabilization.


## The Problem
The main issue is that the model lacks a notion of **confidence** in its predictions. The loss above implicitly assumes $Q_\theta$ has a constant confidence interval, which is unrealistic.  

Intuitively, some game states are much harder to evaluate than others. We therefore need a formulation that accounts for uncertainty.

Moreover, the loss function above does not come from any rigorous first-principles maximum log-likelyhood calculations.


# A First-Principles Approach
Suppose we have a ground truth probability $p^*(Q|s,a)$ of $Q^*$. The **Most likely approximation** $q(Q|s,a)$ of $p^*(Q|s,a)$ given the datapoint $\{s,a,r,s'\}\sim D$ is the one that minimizes the  **negative log-likelihood**:


$$
q = \argmin_q\left\{ -\mathbb{E}_{\{s,a,r,s'\}\sim D} \Big[ \mathbb{E}_{Q^*\sim p^*}\,\log q \Big]\right\}
$$

It can also be expressed as
$$

   q= \argmin_q\left\{-\mathbb{E}_{\{s,a,r,s'\}\sim D} \Big[ \mathrm{KL}(p^*\|q) + H(p^*) \Big]\right\},
$$
where $H(p^*)$ is the entropy of $p^*$.


## Choosing $p$ 
$p^*(Q|s,a)$ is the ground truth, therefore it must respect the Bellman equation:

$$
p^*(Q|s,a) = r(s,a) + \gamma \max_{a'} p^*(Q|s',a'),
$$

> [!Note]
> by $\max p$ i mean the probability distribuition of the biggest $Q$ sampled from each $p(Q|s,a)$. I've decided to use this slightly wrong notation because doing otherwise would make the math needlessly cumbersome

The family of distributions to witch $p^*$ belongs must be **closed under maximization** (for the $\max_{a'}$ term) and **closed under linear transformations** (for the shift by $r$ and scaling by $\gamma$).

These properties hold for the [Generalized Extreme Value (GEV) distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution).  
In this work, we use the [**Gumbel distribution**](https://en.wikipedia.org/wiki/Gumbel_distribution):

$$
\mathrm{Gumbel}(Q|\mu,\beta) 
= \frac{1}{\beta} e^{-(z+e^{-z})}, 
\quad z = \frac{Q-\mu}{\beta}.
$$

Here:

- $\mu = \mu(s,a)$ depends on both state and action,  
- $\beta = \beta(s)$ depends only on the state.  

Knowing that $p$ must be a Gumbel distribution is useful, but we still have the problem that in order to sample from $p(s,a)$ we need to know $p(s',a')$. It's a bit of a chicken and the egg problem.


## Learning a probability distribution $q$
We want to learn a probability function $q(Q|s,a)$ that approximates $p^*$. Since $p^*$ is a Gumbel, this means that all we have to do is to learn the correct $\mu_q=\mu_\theta(s,a)$ and $\beta_q=\beta_\phi(s)$ that depend from some learnable parameters $\theta, \phi$ that describe the whole probability distribution space.

$$
q (Q|s,a)=\textrm {Gumbel}(Q|\mu_q,\beta_q)
$$

As for the target, since we don't know $p^*$ we approximate it with $p$ like so:

$$
p(Q|s,a) = r + \gamma \max_{a'} q(Q|s',a'),
$$


It can be shown that $p$ is equal to:

$$
p(Q|s,a) = \mathrm{Gumbel}(Q|\mu_p,\beta_p),
$$

with

$$
\beta_p = \gamma \cdot \beta_\phi(s'), \quad
\mu_p = r + \gamma \cdot \beta_\phi(s') \cdot 
\log\!\left[ \sum_{a'} \exp\frac{\mu_\theta(s',a')}{\beta_\phi(s')} \right].
$$


With this we can now express analytically the formulas for 
$$
\textrm{KL}(p||q)\quad \textrm{and}\quad H(p)
$$


## Formulas for KL and Entorpy of Gumbel distributions

For $x \sim \mathrm{Gumbel}(\mu,\beta)$:

$$
H(x) = \ln \beta + \gamma_e + 1,
$$

where $\gamma_e \approx 0.5772$ is the Euler–Mascheroni constant.


### KL Divergence Between Two Gumbels

Let $p = \mathrm{Gumbel}(\mu_p,\beta_p)$ and $q = \mathrm{Gumbel}(\mu_q,\beta_q)$.  
The KL divergence has a closed form [[source](https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf)]:

$$
\mathrm{KL}[p\|q] =
\ln\frac{\beta_q}{\beta_p}
+ \frac{\mu_p - \mu_q}{\beta_q}
+ \gamma_e\left(\frac{\beta_p}{\beta_q} - 1\right)
+ \exp\left(-\frac{\mu_p-\mu_q}{\beta_q}\right)
\Gamma\left(1 + \frac{\beta_p}{\beta_q}\right) - 1
$$

where $\Gamma(\cdot)$ is the gamma function.


# The Final Loss
> [Warning]
> The formula might seem scary, but they are actally simple and numerically stable, nothing to be scared of

To improve numerical stability, we reparameterize with $\nu = \log \beta$.  

Then:

$$
\mathrm{KL}[p\|q] =
\nu_q - \nu_p
- (\mu_q-\mu_p)\,e^{-\nu_q} 
+ \gamma_e \big(e^{\nu_p-\nu_q}-1\big) 
+ \exp\!\big[(\mu_q-\mu_p)e^{-\nu_q}\big]\,
\Gamma\!\left(e^{\nu_p-\nu_q}+1\right) - 1
$$

and

$$
H[p] = \nu_p + \gamma_e + 1.
$$

So the total loss becomes:

$$
L = \mathrm{KL}[p\|q] + H[p] =
\nu_q \;-\; (\mu_q-\mu_p)\,e^{-\nu_q}
\;+\;\gamma_e\,e^{\nu_p-\nu_q}
\;+\;\exp\!\big[(\mu_q-\mu_p)e^{-\nu_q}\big]\,
\Gamma\!\big(1 + e^{\nu_p-\nu_q}\big)
$$

If you are at the last move the loss is just equal to 
$$
L = -\log q(r|\mu_q,\beta_q)=\nu_q -(\mu_q-r)e^{-\nu_q} + \exp\!\big[(\mu_q-r)e^{-\nu_q}\big]\,
$$


## Interpretation

- The **KL divergence** measures how much information the model gains when predicting future outcomes.  
- The **Entropy** measures how uncertain the target distribution is.  
- The **total loss** reflects both the **uncertainty** and the **error** of the model when deciding on an action.  


## Final remarks
For stability reasons you need to calculate the gradient over $p$ as well