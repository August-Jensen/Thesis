# The Hamilton Filter

# The Steps of the Algorithm 
### The distribution of $x_t$ is given by
$$f(x_t|X_{0:t-1})=\sum_{i=1}^Nf(x_t|s_t=i)P(s_t=i|X_{0:t-1})$$

### The likeliood contribution
$$f(x_t|x_{t=1}, s_t=i) = \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\left\{-\frac{y^2}{2 \sigma_i^2}\right\}$$

### The Predicition Step:
$$P(s_t=i|X_{0:t-1})=\sum_{j=i}^Np_{ji} P(s_{t-1}=j|X_{0:t-1})$$

### The Filtering Step:
$$P(s_t=i|X_{0:t})=\frac{f(x_t|s_t=i)P(s_t=i|X_{0:t-1})}{\sum_{j=1}^Nf(x_t|s_t=j)P(s_t=j|X_{0:t-1})}$$


### The Smoothing Step:
$$P(s_t=i|X_{0:T};\theta)= \frac{b_t(i)a_t(i)}{\sum_{j=1}^Nb_t(j)a_t(j)}$$
where $a_t(j) = \sum_{i=1}^Nf(x_t|s_t=j;\theta)p_{ij} a_{t-1}(i)$ 


and  $b_t(i) = \sum_{j=1}^Nf(x_{t+1}|s_{t+1}=j; \theta) b_{t+1}(j)p_{ij}$

# THe Algorithm for N state



### The distribution of $x_t$ is given by
$$f(x_t|X_{0:t-1})=\sum_{i=1}^Nf(x_t|s_t=i)P(s_t=i|X_{0:t-1})$$

### The likeliood contribution
$$f(x_t|x_{t=1}, s_t=i) = \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\left\{-\frac{y^2}{2 \sigma_i^2}\right\}$$

### The Steps Step:
$$\begin{equation}
    \hat{\xi}_{t+1|t}=\Pi \hat{\xi}_{t|t}            
\end{equation}$$



$$\begin{equation}
    \hat{\xi}_{t]t}=\frac{(\hat{\xi}{t|t-1}\odot \eta_t)}{\mathbf{1}'(\hat{\xi}_{t|t-1}\odot \eta_t)}            
\end{equation}$$


where $\mathbf{1}\prime$ is an $(N\times 1)$ vector of ones. $\hat{\xi}_{t|t}$ the $(N\times 1)$ vector of filtered probabilities, $\hat{\xi}_{t+1|t}$ the predicted probabilities. $\eta_t$ the $(N\times 1)$ vector if density of $Y_t$ conditional on the past, and, for the N states. $\odot$ is Element by element multiplication, and the parameter values are $\theta$. We take $\hat{\xi}_{1|0}$ as given.\\
$$\eta_t = \begin{bmatrix} f(Y_t|Y_{t-1},s_t=1)\\ \vdots \\ f(Y_t|y_{t-1},s_{t=1} ) \end{bmatrix}$$

While the smoothing step is 
        $$\begin{equation} \hat{\xi}_{t|T}=\hat{\xi}_{t|t}\odot \left[ \Pi' \left[\hat{\xi}_{t+1|T} \oslash \hat{\xi}_{t+1|t} \right]\right] \end{equation}$$
        
with $\oslash$ element by element division, from $t=T$ to $t=0$. The Smoothed probability is iterated backwards, starting at $t=T$.





# The Steps of the Algorithm 
### The distribution of $x_t$ is given by
$$f(x_t|X_{0:t-1})=\sum_{i=1}^Nf(x_t|s_t=i)P(s_t=i|X_{0:t-1})$$

### The likeliood contribution
$$f(x_t|x_{t=1}, s_t=i) = \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\left\{-\frac{y^2}{2 \sigma_i^2}\right\}$$

### The Predicition Step:
$$P(s_t=i|X_{0:t-1})=\sum_{j=i}^Np_{ji} P(s_{t-1}=j|X_{0:t-1})$$

### The Filtering Step:
$$P(s_t=i|X_{0:t})=\frac{f(x_t|s_t=i)P(s_t=i|X_{0:t-1})}{\sum_{j=1}^Nf(x_t|s_t=j)P(s_t=j|X_{0:t-1})}$$


### The Smoothing Step:
$$P(s_t=i|X_{0:T};\theta)= \frac{b_t(i)a_t(i)}{\sum_{j=1}^Nb_t(j)a_t(j)}$$
where $a_t(j) = \sum_{i=1}^Nf(x_t|s_t=j;\theta)p_{ij} a_{t-1}(i)$ 


and  $b_t(i) = \sum_{j=1}^Nf(x_{t+1}|s_{t+1}=j; \theta) b_{t+1}(j)p_{ij}$

# THe Algorithm for N states


        



        
        By iteration.\\
### The distribution of $x_t$ is given by
$$f(x_t|X_{0:t-1})=\sum_{i=1}^Nf(x_t|s_t=i)P(s_t=i|X_{0:t-1})$$

### The likeliood contribution
$$f(x_t|x_{t=1}, s_t=i) = \frac{1}{\sqrt{2\pi \sigma_i^2}}\exp\left\{-\frac{y^2}{2 \sigma_i^2}\right\}$$

### The Steps Step:
$$\begin{equation}
    \hat{\xi}_{t+1|t}=\Pi \hat{\xi}_{t|t}            
\end{equation}$$



$$\begin{equation}
    \hat{\xi}_{t]t}=\frac{(\hat{\xi}{t|t-1}\odot \eta_t)}{\mathbf{1}'(\hat{\xi}_{t|t-1}\odot \eta_t)}            
\end{equation}$$


where $\mathbf{1}\prime$ is an $(N\times 1)$ vector of ones. $\hat{\xi}_{t|t}$ the $(N\times 1)$ vector of filtered probabilities, $\hat{\xi}_{t+1|t}$ the predicted probabilities. $\eta_t$ the $(N\times 1)$ vector if density of $Y_t$ conditional on the past, and, for the N states. $\odot$ is Element by element multiplication, and the parameter values are $\theta$. We take $\hat{\xi}_{1|0}$ as given.\\
$$\eta_t = \begin{bmatrix} f(Y_t|Y_{t-1},s_t=1)\\ \vdots \\ f(Y_t|y_{t-1},s_{t=1} ) \end{bmatrix}$$

While the smoothing step is 
        $$\begin{equation} \hat{\xi}_{t|T}=\hat{\xi}_{t|t}\odot \left[ \Pi' \left[\hat{\xi}_{t+1|T} \oslash \hat{\xi}_{t+1|t} \right]\right] \end{equation}$$
        
with $\oslash$ element by element division, from $t=T$ to $t=0$. The Smoothed probability is iterated backwards, starting at $t=T$.





