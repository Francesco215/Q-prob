# Some Background

Off-policy reinforcement learning revolves around the **Bellman equation**:

$$
Q^*(s,a) = r(s,a) + \gamma \max_{a'} Q^*(s',a')
$$

Where:

- $Q^*(s,a)$ is the optimal $Q$-value for taking action $a$ in state $s$. It represents the maximum total discounted reward the agent can expect from this point forward.
- $r(s,a)$ is the immediate reward received after taking action $a$ from state $s$.
- $\gamma \in (0,1]$ is the discount factor.
- $\max_{a'}Q^*(s',a')$ is the maximum $Q$-value for the next state $s'$ across all possible actions $a'$. This term links the current $Q$-value to the optimal future value, assuming the agent acts optimally.


## The Loss Function

Training is typically **bootstrapped**. A common approach is to minimize the $L_2$ loss:

$$
\mathbb{E}_{\{s,a,r,s'\}\sim D}\left[ Q_\theta(s,a) - \left( r + \gamma \max_{a'} Q_{\bar\theta}(s',a') \right) \right]^2
$$

where $D$ is the replay buffer, $\theta$ are the learnable parameters, and $\bar\theta$ is a delayed copy of the network used for stabilization.


## The Problem

This approach does not scale well to complex, long-horizon tasks.  

The main issue is that the model lacks a notion of **confidence** in its predictions. The loss above implicitly assumes $Q_\theta$ has a constant confidence interval, which is unrealistic.  

Intuitively, some game states are much harder to evaluate than others. We therefore need a formulation that accounts for uncertainty.


# A Probabilistic Approach

Let us assume:

- $Q$ is drawn from a probability distribution $p = p(Q|s,a)$  
- The target is drawn from $q = q(Q|r,s')$, where

$$
q(Q|r,s') = r + \gamma \max_{a'} p(Q|s',a').
$$

We can define the loss as the **negative log-likelihood**:

$$
L = -\mathbb{E}_{\{s,a,r,s'\}\sim D} \Big[ \mathbb{E}_{Q\sim q}\,\log p \Big]
   = -\mathbb{E}_{\{s,a,r,s'\}\sim D} \Big[ \mathrm{KL}(q\|p) + H[q] \Big],
$$

where $H[q]$ is the entropy of $q$.


## Choosing $p$ and $q$

Since

$$
q(Q|r,s') = r + \gamma \max_{a'} p(Q|s',a'),
$$

the family of distributions must be **closed under maximization** (for the $\max_{a'}$ term) and **closed under linear transformations** (for the shift by $r$ and scaling by $\gamma$).

These properties hold for the [Generalized Extreme Value (GEV) distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution).  
In this work, we use the [**Gumbel distribution**](https://en.wikipedia.org/wiki/Gumbel_distribution):

$$
p(Q|s,a) = \mathrm{Gumbel}(Q|\mu,\beta) 
= \frac{1}{\beta} e^{-(z+e^{-z})}, 
\quad z = \frac{Q-\mu}{\beta}.
$$

Here:

- $\mu_p = \mu_\theta(s,a)$ depends on both state and action,  
- $\beta_p = \beta_\phi(s)$ depends only on the state.  

Both are learnable functions parameterized by $\theta$ and $\phi$.


## Target Distribution

It can be shown that under this definition:

$$
q(Q|r,s') = \mathrm{Gumbel}(Q|\mu_q,\beta_q),
$$

with

$$
\beta_q = \gamma \cdot \beta_\phi(s'), \quad
\mu_q = r + \gamma \cdot \beta_\phi(s') \cdot 
\log\!\left[ \sum_{a'} \exp\frac{\mu_\theta(s',a')}{\beta_\phi(s')} \right].
$$


## Useful Properties

### Entropy of a Gumbel

For $X \sim \mathrm{Gumbel}(\mu,\beta)$:

$$
H[X] = \ln \beta + \gamma_e + 1,
$$

where $\gamma_e \approx 0.5772$ is the Euler–Mascheroni constant.


### KL Divergence Between Two Gumbels

Let $p = \mathrm{Gumbel}(\mu_p,\beta_p)$ and $q = \mathrm{Gumbel}(\mu_q,\beta_q)$.  
The KL divergence has a closed form [[source](https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf)]:

$$
\mathrm{KL}[q\|p] =
\ln\frac{\beta_p}{\beta_q}
+ \frac{\mu_q - \mu_p}{\beta_p}
+ \gamma_e\left(\frac{\beta_q}{\beta_p} - 1\right)
+ \exp\left(-\frac{\mu_q-\mu_p}{\beta_p}\right)
\Gamma\left(1 + \frac{\beta_q}{\beta_p}\right) - 1
$$

where $\Gamma(\cdot)$ is the gamma function.


# The Final Loss

To improve numerical stability, we reparameterize with $\nu = \log \beta$.  

Then:

$$
\mathrm{KL}[q\|p] =
\nu_p - \nu_q
- (\mu_p-\mu_q)\,e^{-\nu_p} 
+ \gamma_e \big(e^{\nu_q-\nu_p}-1\big) 
+ \exp\!\big[(\mu_p-\mu_q)e^{-\nu_p}\big]\,
\Gamma\!\left(e^{\nu_q-\nu_p}+1\right) - 1
$$

and

$$
H[q] = \nu_q + \gamma_e + 1.
$$

So the total loss becomes:

$$
L = \mathrm{KL}[q\|p] + H[q] =
\nu_p \;-\; (\mu_p-\mu_q)\,e^{-\nu_p}
\;+\;\gamma_e\,e^{\nu_q-\nu_p}
\;+\;\exp\!\big[(\mu_p-\mu_q)e^{-\nu_p}\big]\,
\Gamma\!\big(1 + e^{\nu_q-\nu_p}\big)
$$


## Interpretation

- The **KL divergence** measures how much information the model gains when predicting future outcomes.  
- The **Entropy** measures how much uncertain the model is when choosing the future move.  
- The **total loss** reflects both the **uncertainty** and the **error** of the model when deciding on an action.  


