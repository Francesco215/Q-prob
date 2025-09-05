# Some background

Off policy reinforcment learning revolves around the bellman equation

$$
Q^*(s,a) = r(s,a) + \gamma\max_{a'}Q^*(s',a')
$$
Where:

- $Q^*(s,a)$ is the optimal $Q$-value for taking action $a$ in state $s$. It represents the maximum total discounted reward the agent can expect to get from this point forward.

- $R(s,a)$ is the immediate reward received after taking action a from state $s$.

- $\gamma\approx1$ is the discount factor.

- $\max_{a'}Q^*(s',a')$ is the maximum $Q$-value for the next state, $s'$, across all possible actions, $a'$. This is the crucial part that links the current $Q$-value to the optimal $Q$-value of the next state, assuming the agent will choose the best possible action from that point on.

## The loss function
The training is usually bootstrapped 

The way is done usually is by minimizing the $L_2$ loss

$$\mathbb E_{\{s,a,r,s'\}\sim D}\left[Q_\theta(s,a) - \left(r + \gamma \max_{a'}Q_{\bar\theta}(s',a') \right) \right]^2$$



## What's the problem here?

The problem is that doesn't work. $Q$-learning, has a major scalability problem when applied to complex, long-horizon tasks. 

There are multiple reasons, but the main one is that the model needs to have a way of estimating the confidence of his prediciton. The loss above assumes that $Q_\theta$ has a constant confidence interval.

Intuitively this is impossible. Some game states can be way more hard to evaluate than others, so how can we fix this?

# The proper way

Lets assume that $Q$ comes from a probability distribuition $p=p(Q|s,a)$ and the target comes from the probability $q=q(Q|r,s')$ where the target is equal to

$$
    q(Q|r,s') = r + \gamma \max_{a'}p(Q|s',a')
$$

The loss is equal to the negative log-likelyhood
$$
L=-\mathbb E_{\{s,a,r,s'\}\sim D}\big[\mathbb E_{Q\sim q}\log p\big] = -\mathbb E_{\{s,a,r,s'\}\sim D}\Big[\textrm{KL}(p||q) + S(q)\Big]
$$

## What are $p$ and $q$?
Since 
$q(Q|r,s') = r + \gamma \max_{a'}p(Q|s',a')$
the family of probability distribuition to wich $p$ and $q$ belong must be closed under maximization (for the $\max_{a'}$ over independent draws from $p$) as well as linear transformations (for the shift by $r$ and scaling by $\gamma$).


The functions are known and belong to [the Generalized extreme value (GEV) distribution family](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution). In this work we are going to use the [Gumbel Distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)

$$
p(Q|s,a)=\textrm{Gumbel}(Q|\mu,\beta)= \frac 1\beta e^{-(z+e^{-z})}\quad\quad \textrm{with } z=\frac {Q-\mu}\beta
$$
Where $\mu$ depends form the state and the action, but $\beta$ just from the state. Both of them from learnable parameters $\theta$ and $\phi$ 
$$\mu = \mu_\theta(s,a)\quad\textrm{and}\quad\beta = \beta_\phi(s)$$

It is possible to show that given this definition of $p$ we have that $q$ is equal to
$$
    q(Q|r,s')=\textrm{Gumbel}(Q|\mu_q,\beta_q)
$$
Where
$$\beta_q = \gamma \cdot \beta_{\phi}(s')$$
$$
\mu_q = r+\gamma\cdot\beta_\phi\log\left[
    \sum_{a'}\exp\frac {\mu_\theta(s',a')}{\beta_\phi(s')}
    \right]
$$

### Entropy of a Gumbel distribution

For $X \sim \mathrm{Gumbel}(\mu,\beta)$, the entropy has a closed form:

$$
H[X] = \ln \beta + \gamma_e + 1
$$

where $\gamma_e \approx 0.5772$ is the Euler–Mascheroni constant.

### KL divergence between two Gumbels

Let $p=\mathrm{Gumbel}(\mu_p,\beta_p)$ and $q=\mathrm{Gumbel}(\mu_q,\beta_q)$.


Then the KL divergence has a closed form [[source](https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf)]:

$$
\mathrm{KL}[p\|q] =
\ln \frac{\beta_q}{\beta_p}
+ \frac{\mu_p - \mu_q}{\beta_q}
+ \gamma_e \left(\frac{\beta_p}{\beta_q} - 1\right)
+ \exp\!\left(-\frac{\mu_p - \mu_q}{\beta_q}\right)\,\Gamma\left(1+\frac{\beta_p}{\beta_q}\right)
- 1,
$$

where $\Gamma(\cdot)$ is the gamma function.


## The loss function

To make the loss function stable we are going to calculate directly $\nu = \log \beta$ 

$$
\textrm{KL}[p||q]=\nu_q-\nu_p + (\mu_p-\mu_q)e^{-\nu_q} + \gamma_e(e^{\nu_p-\nu_q}-1)+\exp\left[-(\mu_p-\mu_q)e^{-\nu_q}\right]\Gamma\left(e^{\nu_p-\nu_q}+1\right) - 1
$$

$$
S(p) = \nu_p + \gamma_e +1
$$

So the total loss is
$$
L = \nu_q + (\mu_p-\mu_q)e^{-\nu_q} + \gamma_ee^{\nu_p-\nu_q}+\exp\left[-(\mu_p-\mu_q)e^{-\nu_q}\right]\Gamma\left(e^{\nu_p-\nu_q}+1\right)
$$

It will be convenient to plot both. The KL divergence serves as a metric that shows us how much information the model can learn from the prediction of his future steps. The total loss is a measure on how uncertain and wrong the model is in general when it comes to choose a good move.

