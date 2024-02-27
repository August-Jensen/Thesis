# The RSDC Model
$x_t=\varepsilon_t$, where both are $(N \times 1)$ vectors. 
For N timeseries. The shock term is modeled as 
$$\varepsilon_t = H_t^{\frac{1}{2}}z_t$$
with $H_t$ an $(N\times N)$ is the conditional covariance matrix. $H_t^{\frac{1}{2}}$ must be positive definite.  $z_t$ is a $(N \times 1)$  vector.

We construct the covariance matrix by
$$H_t=D_tR_tD_t$$
Where $D_t = diag(h_t)$ where $h_t$ is the standard deviations of the univariate ARMACH(1,1), a GARCH(1,1) model in absolute terms rather than squared terms. $R_t$ is the regime switching conditional correlation matrix. For 2 timeseries, the matrix in state i is given by
$$
R_i = 
\begin{pmatrix}1 & \rho_{i}\\
 \rho_{i} & 1
\end{pmatrix}
$$

This describtion originateswas from a GARCH(1,1) model, while the paper i base it on uses ARMACH(1,1), so there may be some errors i have not found. The covariance matrix should be given by 
$$
H_t = 
\begin{pmatrix}h_{1t} & \rho_{s_t}\sqrt{h_{1t}h_{2t}}\\
 \rho_{s_t}\sqrt{h_{1t}h_{2t}} & h_{2t}
\end{pmatrix}
$$ 
An with $z_t=D_t^{-1}\varepsilon_t$ we get the log likelihood contribution to be 
$$log\left(f(x_t|x_{t=1}, s_t=i)\right) = -\frac{1}{2} \left[N log (2 \pi) + 2 log(|D_t|) + log(|R_t|) + z_t\prime R_t^{-1} z_t \right]$$


# The Hamilton filter
Let $\Pi$ be a $(K\times K)$ matrix of transition probabilities, where $p_{ij}=P(s_t=j|s_{t-1}=i)$
Then we can use the law of total probability to find the distriburion of $x_t$.

### The distribution of $x_t$ is given by
$$f(x_t|X_{0:t-1})=\sum_{i=1}^Kf(x_t|s_t=i)P(s_t=i|X_{0:t-1})$$

### The Likelihood Contributions
The simplest model to apply is the Stochastiv Volatility model, with the log likleihood contribution:
$$f(x_t|x_{t=1}, s_t=i) = \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\{-\frac{y^2}{2 \sigma_i^2}\}$$

While the main focus is on the RSDC model,
$$f(x_t|x_{t=1}, s_t=i) =exp\left\{-\frac{1}{2} \left[N log (2 \pi) + 2 log(|D_t|) + log(|R_t|) + z_t\prime R_t^{-1} z_t \right]\right\}$$

We can collect these to a vector form, to perform the steps of the Hamilton filter
$$\eta_t = \begin{bmatrix} f(x_t|x_{t-1},s_t=1)\\ \vdots \\ f(x_t|x_{t-1},s_t=1 ) \end{bmatrix}$$
Where $\eta_t$ is the $(K\times 1)$ vector of the density of $x_t$ conditional on the past, and, for the K states.
### The Predicition Step:
$$P(s_t=i|X_{0:t-1})=\sum_{j=i}^Kp_{ji} P(s_{t-1}=j|X_{0:t-1})$$
Or using the vector form, the predicted probability of for  $t+1$, is the filtered probability at $t$ multiplied by the transition matrix:
$$\begin{equation}
    \hat{\xi}_{t+1|t}=\Pi \hat{\xi}_{t|t}            
\end{equation}$$
Where $\hat{\xi}_{t|t}$ the $(K\times 1)$ vector of filtered probabilities and $\hat{\xi}_{t+1|t}$  is the  $(K\times 1)$ vector of predicted probabilities. 
### The Filtering Step:
$$P(s_t=i|X_{0:t})=\frac{f(x_t|s_t=i)P(s_t=i|X_{0:t-1})}{\sum_{j=1}^Kf(x_t|s_t=j)P(s_t=j|X_{0:t-1})}$$
Or using the vector form:
$$\begin{equation}
    \hat{\xi}_{t]t}=\frac{(\hat{\xi}{t|t-1}\odot \eta_t)}{\mathbf{1}'(\hat{\xi}_{t|t-1}\odot \eta_t)}            
\end{equation}$$
where $\mathbf{1}$ is an $(K\times 1)$ vector of ones. 

 $\odot$ is Element by element multiplication, and the parameter values are $\theta$. We take $\hat{\xi}_{1|0}$ as given.

### The Smoothing Step:
$$P(s_t=i|X_{0:T};\theta)= \frac{b_t(i)a_t(i)}{\sum_{j=1}^NKb_t(j)a_t(j)}$$
where $a_t(j) = \sum_{i=1}^Kf(x_t|s_t=j;\theta)p_{ij} a_{t-1}(i)$ 


and  $b_t(i) = \sum_{j=1}^Kf(x_{t+1}|s_{t+1}=j; \theta) b_{t+1}(j)p_{ij}$
or
$$\begin{equation} \hat{\xi}_{t|T}=\hat{\xi}_{t|t}\odot \left[ \Pi' \left[\hat{\xi}_{t+1|T} \oslash \hat{\xi}_{t+1|t} \right]\right] \end{equation}$$
        
with $\oslash$ element by element division, from $t=T$ to $t=0$. The Smoothed probability is iterated backwards, starting at $t=T$.





