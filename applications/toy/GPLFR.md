2026-01-15
Type: #idea 
Topics: 
References: 

---
# GPLFR

## Main

**Notation:** we consider regression from $D_x$-dimensional inputs to $D_y$-dimensional structured outputs with $N$ training examples.
Let $\mathbf{X}=\begin{bmatrix} \mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_N^\top \end{bmatrix}\in\mathbb{R}^{N\times D_x}$ denote the training inputs
and $\mathbf{Y}=\begin{bmatrix} \mathbf{y}_1^\top \\ \vdots \\ \mathbf{y}_N^\top \end{bmatrix}\in\mathbb{R}^{N\times D_y}$ the corresponding outputs.
GPLFR introduces $D_z\ll D_y$ latent variables per example, collected as
$\mathbf{Z}=\begin{bmatrix} \mathbf{z}_1^\top \\ \vdots \\ \mathbf{z}_N^\top \end{bmatrix} \in\mathbb{R}^{N\times D_z}$, and a linear decoder
$\mathbf{W}\in\mathbb{R}^{D_y\times D_z}$. We write $\mathbf{K}(\mathbf{X}, \mathbf{X})\in\mathbb{R}^{N\times N}$ for the kernel matrix and $\sigma^2$ for the observation noise variance.

### The linear model of coregionalization (LMC) 

The LMC is a very general multi-output GP class that expresses output covariance as a sum of Kronecker products between input kernels $\mathbf{K}_q$ and output covariance ("coregionalization") matrices $\mathbf{B}_q$: $$\text{Cov}(\text{vec}(\mathbf{Y}))=\sum_{q=1}^Q \mathbf{B}_q \otimes \mathbf{K}_q+\sigma^2 \mathbf{I}_{N D_y}, \qquad \mathbf{B}_q \in \mathbb{R}^{D_y\times D_y}, \mathbf{K}_q \in \mathbb{R}^{N\times N}$$ where $\text{vec}(\bullet)$ stacks columns. 
One way to derive this is from a linear mixing of latent GP functions: for each component $q$, draw $D_q$ independent latent functions from a shared kernel $k_q$ and mix them into outputs with a matrix $\mathbf{A}_q\in \mathbb{R}^{D_y\times D_q}$. Then $\mathbf{B}_q=\mathbf{A}_q \mathbf{A}_q^\top\succeq0$ and has rank $\leq D_q.$ 
The intrinsic coregionalized model (ICM) is the $Q=1$ special case: $$\text{Cov}(\text{vec}(\mathbf{Y}))=\mathbf{B}\otimes \mathbf{K} + \sigma^2 \mathbf{I}_{ND_y}.$$See [[@alvarezKernelsVectorValued2012]] for a full taxonomy.
 
### GPLFR and LMC as two marginalizations of the same joint model

We can see the relationship between GPLFR and LMC by starting with their shared assumed data-generating process.
Partition the latent space into $Q$ groups with dimensionalities $\{D_q\}_{q=1}^Q$ such that $\sum_q D_q=D_z$. 
Let $\mathbf{Z}_q\in \mathbb{R}^{N\times D_q}$ and $\mathbf{W}_q\in \mathbb{R}^{D_y\times D_q}$, and define $\mathbf{Z}=\begin{bmatrix} \mathbf{Z}_1 & \dots & \mathbf{Z}_Q \end{bmatrix},\ \mathbf{W}=\begin{bmatrix} \mathbf{W}_1 & \dots & \mathbf{W}_Q \end{bmatrix}$ 
The data-generating process draws the latent components from independent GP priors over $\mathbf{X}$, and then maps them through a linear-Gaussian decoder: $$\begin{align}\text{latent GP priors}&:\quad\text{vec}(\mathbf{Z}_q) \mid \mathbf{X} \sim  \mathcal{N}(0, \mathbf{I}_{D_q}\otimes \mathbf{K}_q)\quad \text{for}\quad q=1,...,Q \\ \text{decoder} &: \quad \mathbf{Y}=\sum_{q} \mathbf{Z}_q \mathbf{W}_q^\top + \mathbf{E}, \quad \text{where}\quad E_{ij}\sim \mathcal{N}(0,\sigma^2) \end{align}$$
If we marginalize out $\mathbf{Z}$ (fitting $\mathbf{W}$ later), then we get the marginal $$\begin{align} p(\text{vec}(\mathbf{Y}) \mid \{\mathbf{W}_q\}_q, \sigma, \mathbf{X})&=\int p(\text{vec}(\mathbf{Y})\mid \{\mathbf{Z}_q\}_q, \{\mathbf{W}_q\}_q, \sigma) \prod_q p(\mathbf{Z}_q \mid \mathbf{X})\,d \mathbf{Z} \\&=\mathcal{N}(\text{vec}(\mathbf{Y}); 0, \mathbf{C}_\text{LMC})   \end{align}$$with $$\mathbf{C}_\text{LMC}=\left[\sum_q (\mathbf{W}_q \mathbf{W}_q^\top) \otimes \mathbf{K}_q\right]+ \sigma^2\mathbf{I}_{ND_y}.$$ This is exactly an LMC GP with $\mathbf{B}_q=\mathbf{W}_q \mathbf{W}_q^\top$.

In GPLFR we instead marginalize out $\mathbf{W}$ (fitting $\mathbf{Z}$ later). To do this we first set priors on the component decoders: $\mathbf{W}_q\sim \mathcal{MN}(0, \mathbf{B}, \mathbf{T}_{q})$ independently for each $q$, where $\mathbf{B}$ and $\mathbf{T}_{q}$ are the row and column covariances. Then  $$\begin{align} p(\text{vec}(\mathbf{Y}) \mid \{\mathbf{Z}_q\}_q, \sigma)&=\int p(\text{vec}(\mathbf{Y}) \mid \{\mathbf{Z}_q\}_q, \{\mathbf{W}_q\}_q, \sigma) \prod_q p(\mathbf{W}_q)\,d \mathbf{W}\\&=\mathcal{N}(\text{vec}(\mathbf{Y}); 0, \mathbf{C})  \end{align}$$ with $$\mathbf{C} =\mathbf{B}\otimes \left[\sum_q(\mathbf{Z}_q\mathbf{T}_{q} \mathbf{Z}_q^\top)\right]+\sigma^2 \mathbf{I}_{N D_y}.$$This can be interpreted as an ICM covariance conditional on $\mathbf{Z}$, where the effective input-side kernel is that learned sum (if $\mathbf{Z}$ is point-estimated).  

With this perspective, GPLFR and LMC are seen as different marginalizations of the same underlying factorization.
This is a similar idea to the *primal* and *dual* views of probabilitistic PCA used to motivate GPLVMs in [[@lawrenceProbabilisticNonlinear2005]]. However, note that while the primal and dual views are *equivalent* (they recover the same marginal model) in the case of probabilistic PCA, the regression context of GPLFR/LMC ($\mathbf{Z}$ tied to $\mathbf{X}$ by a GP prior) immediately separates them into different model classes.^footnote 
- 🦶 Excepting degenerate cases, e.g., if we restrict GPLFR's latent representation to be deterministic feature maps of the inputs $\mathbf{Z}_q=\boldsymbol{\Phi}_q(\mathbf{X})$ then we get ordinary LMC priors with input kernels $k_q(\mathbf{x}, \mathbf{x}'; \boldsymbol{\Phi})=\boldsymbol{\phi}_q(\mathbf{x})^\top \mathbf{T}_q \boldsymbol{\phi}_q(\mathbf{x}')$. 

### Practical choices in GPLFR

The above formulation is very general, and experimenting with the full generalities is likely worthwile. However, for simplicity in this work we tend to make the following restrictions:
- We take the per-latent GP case: $D_q=1$ for all $q$ (hence $Q=D_z$).  In this case, the decoder column covariance $\mathbf{T}_{q}$ is just a scalar, so the set of different $\mathbf{T}_{q}$'s reduces to a vector of different latent scales $\boldsymbol{\tau}^2\in \mathbb{R}^{D_z}$. This simplifies the $\mathbf{Z}$-conditional output covariance to $$\mathbf{C}=\mathbf{B}\otimes (\mathbf{Z} \text{diag}(\boldsymbol{\tau}^2) \mathbf{Z}^\top) + \sigma^2 \mathbf{I}_{ND_y}$$
	- 🦶even if $D_q >1$, $\mathbf{T}_q$ should not be richly structured since it is only identifiable through the product $\mathbf{Z}_q \mathbf{T}_q \mathbf{Z}_q^\top$. 
- Doing the alternative marginalization ($\mathbf{Z}$) for this case yields the semiparametric latent factor model (SLFM) of [[@tehSemiparametricLatent2005]] with $\mathbf{C}_\text{LMC} =\sum_{q=1}^{D_z} (\mathbf{w}^{(q)} \mathbf{w}^{(q)\top}) \otimes \mathbf{K}_q+ \sigma^2 \mathbf{I}_{ND_y}$  

In practice, we also often restrict the decoder row covariance to $\mathbf{B}=\mathbf{I}$, which simplifies $\mathbf{C}$ to a block diagonal structure; although, if some cross-output structure is both important and cheaply parameterized, this​ can be worth relaxing.


:obs_note_glyph: In using $D_q>1$, it might be worth structuring $\mathbf{T}_q=\text{diag}(\tau^2_{q,1},...,\tau^2_{q, D_q})$, especially for MCMC. Since otherwise we get orthogonal transformation invariance of $\mathbf{Z}_q$. This invariance doesn't affect predictions, but can reduce sample efficiency for MCMC methods and worsen conditioning for MAP estimation. In this case you can still write $\mathbf{C}$  in the $D_q=1$ simplified form where $\mathbf{T}=\text{blockdiag}(\mathbf{T}_1, ..., \mathbf{T}_Q)=\text{diag}(\boldsymbol{\tau}^2)$where $\boldsymbol{\tau}^2$ concatenates all the $\{\tau^2_{q,r}\}$ in the same column order as $\mathbf{Z}$.

## Notation and data layout 

- input $\mathbf{x}\in \mathbb{R}^{D_x}$. The matrix of training inputs is denoted $\mathbf{X}=\begin{bmatrix} \mathbf{x}_1 & ... & \mathbf{x}_N \end{bmatrix}^\top$. 
- output $\mathbf{y}\in \mathbb{R}^{D_y}$.  The matrix of training outputs is denoted $\mathbf{Y}=\begin{bmatrix} \mathbf{y}_1 & ... & \mathbf{y}_N \end{bmatrix}^\top$ 
- latent $\mathbf{z}\in \mathbb{R}^{D_z}$, with associated matrix $\mathbf{Z}=\begin{bmatrix} \mathbf{z}_1 & ... & \mathbf{z}_N \end{bmatrix}^\top$   
- we call anything you infer via MAP or MCMC simply *parameters*. Fixed design choices are referred to as *hyperparameters* and are omitted from conditioning notation. 
- Unless otherwise stated, $\mathbf{X}$ and $\mathbf{Y}$ are z-scored.
- for clarity we provide the GPLFR details only for the *per-latent* GP ($D_q=1\ \forall q$) special case. This is the case used in all the example problems in this paper and extension to grouped-latent GPs is straightforward. 

## GPLFR details 
### Generative model
- **encoder**
	- for each latent dim $q\in {1,...,D_z}$, we introduce a latent function $f_d: \mathbb{R}^{D_x}\to \mathbb{R}$. We give each function an independent zero-mean GP prior $f_q\sim \mathcal{GP}(0, k_q)$. This yields the latent GP prior $$ \mathbb{R}^N\ni\mathbf{z}^{(q)}\mid \mathbf{X}, \boldsymbol{\ell}_q, \eta_q \sim \mathcal{N}(0, \mathbf{K}_q)$$where $K_{q,ij}=k_q(\mathbf{x}_i, \mathbf{x}_j; \boldsymbol{\ell}_q, \eta_q)$. 
	- each kernel is parameterized by lengthscales $\boldsymbol{\ell}_q\in \mathbb{R}^{D_x}$, and amplitude $\eta_q$  
		- :obs_note_glyph:(perhaps) (but not an amplitude, which would be very weakly identified against parameters in the decoder)
	- collecting across $q$ gives $$p(\mathbf{Z} \mid \mathbf{X}, \{\boldsymbol{\ell}_q, \eta_q\}_q)=\prod_{q=1}^{D_z} p(\mathbf{z}^{(q)} \mid \mathbf{X}, \boldsymbol{\ell}_q, \eta_q)$$
- **decoder**
	- we use a linear-Gaussian decoder $$\mathbf{y}\mid \mathbf{z}, \mathbf{W}, \sigma\sim \mathcal{N}(\mathbf{W} \mathbf{z}, \sigma^2 \mathbf{I})$$with weight prior $$\mathbf{W}\sim \mathcal{MN}(0, \mathbf{B}, \text{diag}(\boldsymbol{\tau}^2)).$$
	- for our high-dimension outputs regime, an unconstrained $\mathbf{B}$ is far too weakly identified to learn reliably, even a rank-1 $\mathbf{B}$ is likely to be too weakly identified. The default option should therefore simply be $\mathbf{B}=\mathbf{I}.$ However, if the outputs have an obvious structuring it can be reasonable to relax this. We see an example of this in exoGCM where we can naturally split outputs into different fields - a low-rank covariance across whole fields has a small number of parameters and improves predictions. This is much like the common use case of LMC-style MOGPs of learning output covariance across tasks, except here each task is itself a high-dimensional object.
- **PGM** (see overleaf tikz)

### Model fitting 
- **collapsed decoder likelihood**
	- we marginalize out the decoder parameters $$p(\mathbf{Y}\mid \mathbf{Z},\sigma^2)=\int p(\mathbf{Y} \mid \mathbf{Z}, \mathbf{W}, \sigma^2) p(\mathbf{W} )\, d \mathbf{W} $$
	- for one output dimension $j$ (i.e., a column of $\mathbf{Y}$) $$\mathbf{y}^{(j)} \mid \mathbf{Z}, \mathbf{w}^{(j)}, \sigma^2 \sim \mathcal{N}( \mathbf{Z} \mathbf{w}^{(j)}, \sigma^2 \mathbf{I}_N)$$
	- marginalizing out $\mathbf{W}\sim \mathcal{MN}(0, \mathbf{B}, \text{diag}(\boldsymbol{\tau}^2))$ gives  $$\text{vec}(\mathbf{Y}) \mid \mathbf{Z}, \mathbf{B}, \boldsymbol{\tau}, \sigma \sim \mathcal{N}(0, \mathbf{C}), \quad \mathbf{C}=\mathbf{B}\otimes (\mathbf{Z} \text{diag}(\boldsymbol{\tau}^2)\mathbf{Z}^\top)+\sigma^2 \mathbf{I}_{ND_y}.$$
	- this is the prior predictive.
		- :obs_note_glyph: for $\mathbf{B}=\mathbf{I}$, since all columns are conditionally independent given $\mathbf{Z}$, this reduces to $$p(\mathbf{Y}\mid \mathbf{Z}, \boldsymbol{\tau}, \sigma)=  \prod_{j=1}^{D_y} \mathcal{N}(\mathbf{y}^{(j)}; 0, \mathbf{C}_N), \quad \mathbf{C}= \mathbf{Z} \text{diag}(\boldsymbol{\tau}^2)\mathbf{Z}^\top+\sigma^2 \mathbf{I}_{N}$$ And $$\log p(\mathbf{Y}|\mathbf{Z}, \boldsymbol{\tau},  \sigma)=\sum_j \log \mathcal{N}(\mathbf{y}^{(j)};0, \mathbf{C}_N)$$
		- **efficient computation**
			- whenever we evaluate this distribution, we need to evaluate $\mathbf{C}^{-1}$. For this we use the matrix inversion lemma for a diagonal plus low-rank matrix (Woodbury identity).
			- structured $\mathbf{B}$ case
				- define $\mathbf{V} = \mathbf{Z}\text{diag}(\boldsymbol{\tau}) \in \mathbb{R}^{N\times D_z}$ and $\mathbf{S} = \mathbf{V}^\top \mathbf{V} \in \mathbb{R}^{D_z \times D_z}$.
				- then $\mathbf{C} = \sigma^2 \mathbf{I}_{ND_y} + \mathbf{W}\mathbf{W}^\top$ where $\mathbf{W} = \mathbf{B}^{1/2} \otimes \mathbf{V}$.
				- define $\mathbf{D} = \sigma^2 \mathbf{I}_{D_y D_z} + \mathbf{B} \otimes \mathbf{S}$.
				- by Woodbury identity: $$\mathbf{C}^{-1} = \frac{1}{\sigma^2}\left(\mathbf{I}_{ND_y} - \mathbf{W}\mathbf{D}^{-1}\mathbf{W}^\top\right)$$ $$\log \det \mathbf{C} = (ND_y - D_y D_z)\log \sigma^2 + \log \det \mathbf{D}$$
				- **diagonalizing D**: eigendecompose $\mathbf{B} = \mathbf{U}_B \boldsymbol{\Lambda}_B \mathbf{U}_B^\top$ and $\mathbf{S} = \mathbf{U}_S \boldsymbol{\Lambda}_S \mathbf{U}_S^\top$, then $$\mathbf{D}^{-1} = (\mathbf{U}_B \otimes \mathbf{U}_S)\,\text{diag}\left(\frac{1}{\lambda_i^B \lambda_j^S + \sigma^2}\right)_{ij}\,(\mathbf{U}_B \otimes \mathbf{U}_S)^\top$$ $$\log \det \mathbf{D} = \sum_{i=1}^{D_y}\sum_{j=1}^{D_z} \log(\lambda_i^B \lambda_j^S + \sigma^2)$$
				- **quadratic form**: define $\tilde{\mathbf{Y}} = \mathbf{U}_S^\top \mathbf{V}^\top \mathbf{Y} \mathbf{U}_B \boldsymbol{\Lambda}_B^{1/2} \in \mathbb{R}^{D_z \times D_y}$, then $$\text{vec}(\mathbf{Y})^\top \mathbf{C}^{-1} \text{vec}(\mathbf{Y}) = \frac{1}{\sigma^2}\left(\|\mathbf{Y}\|_F^2 - \sum_{i=1}^{D_y}\sum_{j=1}^{D_z}\frac{\tilde{Y}_{ji}^2}{\lambda_i^B \lambda_j^S + \sigma^2}\right)$$
				- computational complexity is $O(ND_z D_y + D_y^3 + D_z^3)$. the Kronecker structure avoids the naïve $O(N^3 D_y^3)$ cost.
			-  $\mathbf{B}=\mathbf{I}$ case. 
				- define $\mathbf{V} = \mathbf{Z}\text{diag}(\boldsymbol{\tau}) \in \mathbb{R}^{N\times D_z}$, then $\mathbf{C}_N=\sigma^2 \mathbf{I}_N + \mathbf{V} \mathbf{V}^\top$.
				- also define $\mathbf{D}= \sigma^2 \mathbf{I}_{D_z}+ \mathbf{V}^\top \mathbf{V}$.
				- then $$\mathbf{C}_N^{-1}=\frac{1}{\sigma^2}( \mathbf{I}_N- \mathbf{V} \mathbf{D}^{-1}\mathbf{V}^\top)$$and the matrix determinant$$\log \det \mathbf{C}_N=(N-D_z)\log \sigma^2 + \log \det \mathbf{D}$$
				- for $\mathbf{y}=\mathbf{y}^{(j)}\equiv \mathbf{Y}_{:j}$ the quadratic form is $$\mathbf{y}^\top \mathbf{C}_N^{-1} \mathbf{y} = \frac{1}{\sigma^2}\left(\|\mathbf{y}\|_2^2 -(\mathbf{V}^\top \mathbf{y})^\top \mathbf{D}^{-1} (\mathbf{V}^\top \mathbf{y}) \right)$$
				- this makes the computational complexity $O(N D_z D_y + D_z^2 D_y + N D_z^2+D_z^3)$ which is typically dominated by the $O(ND_zD_y)$ term.
- **posterior over latents and parameters** 
	- the posterior over latents and global parameters is $$p(\underbrace{\mathbf{Z}, \mathbf{B},  \boldsymbol{\tau}, \sigma, \{\boldsymbol{\ell}_q, \eta_q\}_q}_\boldsymbol{\phi} \mid \underbrace{\mathbf{Y}, \mathbf{X}}_\mathcal{D})\propto  \underbrace{p(\mathbf{Y} \mid\mathbf{Z}, \mathbf{B},  \boldsymbol{\tau},  \sigma)^\beta}_\text{collapsed decoder likelihood} \underbrace{p(\mathbf{Z} \mid \mathbf{X}, \{\boldsymbol{\ell}_q, \eta_q\}_q )}_{\text{encoder prior}} \, p(\boldsymbol{\tau})\, p(\{\boldsymbol{\ell}_q, \eta_q\}_q)\, p(\sigma^2)$$ 
		- :obs_note_glyph: $p(\mathbf{Y} \mid \mathbf{Z}, \mathbf{B}, \boldsymbol{\tau}, \sigma^2)$ is Gaussian in $\mathbf{Y}$, but not in $\mathbf{Z}$, so no easy sampling
	- **likelihood tempering**
		- We use a tempered likelihood with power $\beta < 1$. This accounts for model misspecification arising from the conditional independence assumption in the decoder, which is unlikely to hold exactly. Under such misspecification, standard maximum likelihood or MAP estimation can lead to overconfident inference as the likelihood overwhelms the prior. The tempering parameter $\beta$ can be interpreted as controlling the effective sample size [[@millerRobustBayesian2015]]. 
		- $\beta$ can be tuned like any other hyperparameter, however we find the heuristic $\beta=D_\text{eff}^*/D_y$, where $D_\text{eff}^*$ is the effective rank of the output correlation matrix (corrected for finite sample size), to reliably give good results.
			- Concretely, for output correlation martix $\mathbf{R}$, the effective rank is $$D_\text{eff} =\frac{D_y^2}{\|\mathbf{R}\|_{F}^2}=\frac{D_y}{1+(D_y-1)\overline{\rho^2}}$$where $\overline{\rho^2}=\frac{1}{D_y(D_y-1)}\sum_{i\neq j}\rho_{ij}^2$ is the mean squared off-diagonal correlation.
			- we correct for finite $N$ by subtracting the sampling noise floor: $\overline{\rho^2}_*=\overline{\rho^2}-\frac{1}{N-1}$, and hence $$\beta=\frac{D_\text{eff}^*}{D_y}=\frac{1}{1+(D_y-1)\overline{\rho^2_*}}\approx \frac{1}{1+D_y \overline{\rho^2}-D_y/N}$$ 
				- (This correction comes from the fact that $\overline{\rho^2}=\frac{1}{N-1}$ if the output dimensions are actually all independent and Gaussian distributed)
		- For some slightly lighter outputs we use a low-rank plus diagonal model for output correlations (i.e., $\mathbf{R}=\mathbf{R}_r \mathbf{R}_r^\top+ \text{diag}(\boldsymbol{\psi})$ where $\mathbf{R}_r$ is rank-$r$). Although this brings the effective rank closer to $D_y$, for small $r$ the effect can still be significant. Therefore, we use the more general heuristic of treating the model as explaining the $r$ largest correlation modes and temper for the remaining correlations: $$ \beta \;\approx\; \frac{r + D_{\mathrm{eff,tail}}^{(r)*}}{D_y}$$where $D_{\mathrm{eff,tail}}^{(r)} \;\equiv\; \frac{\big(\sum_{k=r+1}^{D_y} \lambda_k\big)^2}{\sum_{k=r+1}^{D_y} \lambda_k^2}$ and $D_{\mathrm{eff,tail}}^{(r)*}$ is its finite-$N$-corrected version.
		- One can adjust appropriately for more structured models of output correlations (although tempering becomes less important as the output-correlation model becomes more expressive). 
			- E.g., for ROXCE, we use a factorized tempering $\beta \approx \beta_{\mathrm{field}}\beta_{\mathrm{within}}$ for the field-structured output correlation model. $\beta_{\mathrm{within}}$ is computed from within-field correlations (averaged over fields) using the independence-case estimator above, and $\beta_{\mathrm{field}}$ uses the same rank-1 tail proxy applied to the field–field correlation matrix.
#### Independence case $\mathbf{B}=\mathbf{I}$
- Let $\mathbf{Y}\in\mathbb{R}^{N\times D}$ denote the (train) outputs after per-coordinate standardization (zero mean and unit variance over the training set), and define the sample correlation matrix $$\mathbf{R} \;=\; \frac{1}{N-1}\mathbf{Y}^\top\mathbf{Y}\in\mathbb{R}^{D\times D}.$$
- Define the (squared-Frobenius) second moment $\mathrm{tr}(\mathbf{R}^2)=\|\mathbf{R}\|_F^2$ and debias it by subtracting the i.i.d.-Gaussian sampling noise floor in the off-diagonal correlations: $$\mathrm{tr}(\mathbf{R}^2)_* \;\equiv\; \mathrm{tr}(\mathbf{R}^2)\;-\;\frac{D(D-1)}{N-1}.$$
- We then set $$D_{\mathrm{eff},*}\;\equiv\;\frac{(\mathrm{tr}\,\mathbf{R})^2}{\mathrm{tr}(\mathbf{R}^2)_*}\;=\;\frac{D^2}{\mathrm{tr}(\mathbf{R}^2)_*},\qquad \beta\;\equiv\;\frac{D_{\mathrm{eff},*}}{D}\;=\;\frac{D}{\mathrm{tr}(\mathbf{R}^2)_*}.$$
- In practice we clip $\beta$ to $(0,1]$ (and optionally floor $\mathrm{tr}(\mathbf{R}^2)_*$ to a small $\epsilon>0$ for numerical safety).

#### Rank-$r$ plus diagonal output covariance
- When using a restricted coregionalization $\mathbf{B}=\mathbf{B}_r+\mathrm{diag}(\boldsymbol\psi)$ with $\mathrm{rank}(\mathbf{B}_r)=r$, we use the proxy that the model can explain the leading $r$ global correlation modes of $\mathbf{R}$. Let $\lambda_1\ge\dots\ge\lambda_D$ denote eigenvalues of $\mathbf{R}$.
- Define the debiased tail second moment by removing the leading modes from the debiased $\mathrm{tr}(\mathbf{R}^2)_*$: $$S_{2,\mathrm{tail},*}^{(r)} \;\equiv\; \mathrm{tr}(\mathbf{R}^2)_* \;-\;\sum_{k=1}^{r}\lambda_k^2,$$ and the tail trace $$S_{1,\mathrm{tail}}^{(r)} \;\equiv\; \mathrm{tr}(\mathbf{R}) \;-\;\sum_{k=1}^{r}\lambda_k \;=\; D-\sum_{k=1}^{r}\lambda_k.$$
- The corresponding tail effective rank and tempering are $$D_{\mathrm{eff,tail},*}^{(r)} \;\equiv\; \frac{\big(S_{1,\mathrm{tail}}^{(r)}\big)^2}{S_{2,\mathrm{tail},*}^{(r)}},\qquad \beta \;\approx\; \frac{r + D_{\mathrm{eff,tail},*}^{(r)}}{D},$$ which reduces to the rank-1 expression used in our implementation.

#### Field-structured outputs
- For outputs with field structure $D_y=F\cdot A$, we optionally use $\beta \approx \beta_{\mathrm{field}}\beta_{\mathrm{within}}$.
    - $\beta_{\mathrm{within}}$ is computed by applying the independence-case estimator to each field’s $A\times A$ correlation (averaged over fields).
    - $\beta_{\mathrm{field}}$ applies the rank-1 tail proxy above to an empirical $F\times F$ field–field correlation matrix constructed by stacking coefficient locations as additional samples; the finite-$N$ debiasing uses an effective sample size $N_{\mathrm{eff}}\approx N\,A_{\mathrm{eff}}$ to account for within-field correlations.

 
	- **MAP estimation** 
		- it is possible to approximate this posterior with MCMC and this is a viable route for low-data problems and healthy compute budgets. For this work however, we use a computationally cheaper MAP estimation. I.e., we approximate $p(\boldsymbol{\phi} \mid \mathcal{D})\approx \delta(\boldsymbol{\phi}-\boldsymbol{\phi}^\star)$ where $$\boldsymbol{\phi}^\star=\mathop{\arg\max}_\boldsymbol{\phi} \log p(\boldsymbol{\phi}\mid \mathcal{D})$$
		- this means the **MAP objective** is
			- . $$\begin{align}J &= \beta\log p(\mathbf{Y}\mid \mathbf{Z}, \mathbf{B},\boldsymbol{\tau}, \sigma)+\log  p(\mathbf{Z} \mid \mathbf{X}, \{\boldsymbol{\ell}_q, \eta_q\}_q )+\log p(\mathbf{B})+\log p(\{\boldsymbol{\ell}_q, \eta_q\}_q)+\log p(\boldsymbol{\tau})+\log p(\sigma^2) \\ \\&= - \frac{\beta}{2}\left[\log \det \mathbf{C} +\text{vec}(\mathbf{Y})^\top \mathbf{C}^{-1}\text{vec}(\mathbf{Y})\right] - \frac{1}{2}\left[\sum_q \left( \mathbf{z}^{(q)\top}\mathbf{K}_q^{-1}\mathbf{z}^{(q)} + \log\det \mathbf{K}_q\right)\right] + \begin{array}{c}\text{prior terms for}\\ \text{global parameters}\end{array}   \end{align}$$
			- for $\mathbf{B}=\mathbf{I}_{D_y}$ 
				- the likelihood term reduces to the factorized form $D_y\log \det \mathbf{C}_N +\text{tr}(\mathbf{C}_N^{-1}\mathbf{Y}\mathbf{Y}^\top )$ 
			- :obs_note_glyph: learning $\boldsymbol{\tau}$ or $\boldsymbol{\eta}$ 
				- so $\boldsymbol{\tau}$ is constrained by the decoder (likelihood) term, whereas $\boldsymbol{\eta}$ is constrained by the encoder (prior) term

### prediction (i.e., posterior predictive sampling)
- for a given test input $\mathbf{x}_*$, **the posterior predictive distribution** $$\begin{align} p(\mathbf{y}_* \mid \mathbf{x}_*,\mathcal{D})&=\int p(\mathbf{y}_* \mid \mathbf{x}_*, \mathbf{Y}, \boldsymbol{\phi})   p(\boldsymbol{\phi} \mid \mathcal{D}) d \boldsymbol{\phi} \\&\approx p(\mathbf{y}_* \mid \mathbf{x}_*, \mathbf{Y}, \boldsymbol{\phi}^\star)\qquad \qquad \qquad\text{(MAP approximation)} \end{align}$$
- we sample from this MAP-approximated distribution in two steps, corresponding to the encoder and decoder (we leave the $\bullet^\star$ superscript implicit for brevity): 
	1. **encoder:** sample the test latent from the **posterior predictive over latents** $p(\mathbf{z}_* \mid \mathbf{Z}^\star, \mathbf{X}, \{\boldsymbol{\ell} _q\}_q)$.
		- for dimension $q$ $$z_*^{(q)}\mid \mathbf{z}^{\star(q)},\mathbf{X},\boldsymbol{\ell}_{q},\eta_q \sim \mathcal{N}\!\Big( \underbrace{\mathbf{k}_{*,q}^\top \mathbf{K}_q^{-1}\mathbf{z}^{\star(q)}}_{\mu^{(q)}_{z_*}}, \; \underbrace{k_{**,q}-\mathbf{k}_{*,q}^\top \mathbf{K}_q^{-1}\mathbf{k}_{*,q}}_{{\sigma^{(q)}_{z_*}}^2} \Big).$$where $$K_{q,ij}=k_q\big(\mathbf{x}_i,\mathbf{x}_j;\boldsymbol{\ell}_q, \eta_q\big), \quad \mathbf{k}_{*,q}=k_q\big(\mathbf{x}_*,\mathbf{X};\boldsymbol{\ell}_q, \eta_q\big), \quad k_{**,q}=k_q\big(\mathbf{x}_*, \mathbf{x}_*;\boldsymbol{\ell} _q, \eta_q\big).$$ 
		- Stacking these over $q$ gives $$\mathbf{z}_*\mid \mathbf{Z} ,\mathbf{X},\{\boldsymbol{\ell} _{q}, \eta_q\}_q \sim \mathcal{N}\!\big(\boldsymbol{\mu}_{z_*},\; \mathrm{diag}(\boldsymbol{\sigma}^2_{z_*})\big),$$where $(\boldsymbol{\mu}_{z_*})_q=\mu^{(q)}_{z_*}$ and $(\boldsymbol{\sigma}^2_{z_*})_q={\sigma^{(q)}_{z_*}}^2$. 
	2. **decoder:** sample the test output given the test latent from the **latent-conditional posterior predictive**  $p(\mathbf{y}_* \mid \mathbf{z}_*,\mathbf{Y}, \mathbf{Z} , \mathbf{B} , \boldsymbol{\tau} , \sigma )$   
		- This is the posterior predictive after we've conditioned on the posterior state and the test latent 
		- After marginalizing $\mathbf{W}$, the joint Gaussian over training outputs and the test output (conditioned on $\mathbf{z}_*$) is $$ \begin{bmatrix} \mathrm{vec}(\mathbf{Y}) \\ \mathbf{y}_* \end{bmatrix} \Bigm| \; \mathbf{z}_*,\mathbf{Z} , \mathbf{B} , \boldsymbol{\tau} , \sigma  \sim \mathcal{N}\!\left( \mathbf{0}, \begin{bmatrix} \mathbf{C}_{YY} & \mathbf{C}_{Y*} \\ \mathbf{C}_{*Y} & \mathbf{C}_{**} \end{bmatrix} \right)$$with blocks $\mathbf{C}_{YY}=\mathbf{B}\otimes (\mathbf{Z} \text{diag}(\boldsymbol{\tau}^2)\mathbf{Z}^\top)+\sigma^2 \mathbf{I}_{ND_y},$ $\mathbf{C}_{Y*}=\mathbf{B}\otimes(\mathbf{Z} \text{diag}(\boldsymbol{\tau}^2) \mathbf{z}_*),$ $\mathbf{C}_{**} = \mathbf{B}\otimes(\mathbf{z}_*^\top \text{diag}(\boldsymbol{\tau}^2) \mathbf{z}_*)  + \sigma^2 \mathbf{I}_{D_y}.$ 
		- conditioning on the observed $\mathbf{Y}$ gives $$\mathbf{y}_* \mid \mathbf{z}_*, \mathbf{Y}, \mathbf{Z},\mathbf{B}, \boldsymbol{\tau},\sigma \sim \mathcal{N}(\boldsymbol{\mu}_{y_*}, \boldsymbol{\Sigma}_{y_*})$$ where $$\boldsymbol{\mu}_{y_*}=\mathbf{C}_{*Y} \mathbf{C}_{YY}^{-1} \text{vec}(\mathbf{Y}), \qquad \boldsymbol{\Sigma}_{y_*}= \mathbf{C}_{**}-\mathbf{C}_{*Y}  \mathbf{C}_{YY}^{-1}\mathbf{C}_{Y*}$$
			- note: if $\mathbf{B}=\mathbf{I}_{D_y}$ then $\mathbf{C}_{YY}$ is block-diagonal and the above reduces to independent per-$j$ conditioning with $N\times N$ covariance $\mathbf{Z} \text{diag}(\boldsymbol{\tau}^2)\mathbf{Z}^\top +\sigma^2 \mathbf{I}_{N}$ 
		- :obs_note_glyph: the code operates / is written in a slightly different but *equivalent* way - specifically it is written in a weight-space view whereas the above is written in a data-space view
- with (MAP posterior predictive) samples $\{\mathbf{y}_*^{[m]}\}_{m=1}^{M}$ we can estimate the **posterior predictive mean and covariance**  $$\mathbb{E}(\mathbf{y}_*|\mathcal{D})\approx \hat{\mathbf{y}}_* \equiv\frac{1}{M}\sum_{m} \mathbf{y}_*^{[m]},\quad \text{Cov}(\mathbf{y}_*|\mathcal{D})  \approx \hat{\boldsymbol{\Sigma}}_* \equiv \frac{1}{M-1} \sum_m \left( \mathbf{y}_*^{[m]}-\hat{\mathbf{y}}_*\right) \left( \mathbf{y}_*^{[m]}-\hat{\mathbf{y}}_*\right)^\top$$
	- in practice we compute $\hat{\mathbf{y}}_*$ analytically using the means of encoder and decoder predictive distributions to avoid unnecessary Monte Carlo noise. This also enables much faster prediction if uncertainty is not needed.

## Priors and hyperparameters 

### Encoder 
- for all example problems we use Matern-5/2 ARD kernels for each latent coordinate $q$.  
- we use prior on lengthscales: $\log\ell\sim \mathcal{N}(0, 1)$, ...

### Decoder
- we use prior on weights $\mathbf{W}\sim \mathcal{MN}(0, \mathbf{B}, \text{diag}(\boldsymbol{\tau}^2))$
	- $\boldsymbol{\tau}$: for all problems we use the prior: $\log\boldsymbol{\tau}\sim \mathcal{N}(0,0.5^2)$. We then impose that $\boldsymbol{\tau}$ has unit geometric mean. (for now we fix overall scale but maybe can learn it)
	- $\mathbf{B}$: 
		- **Phoenix experiment:** $\mathbf{B}=\mathbf{I}$ 
		- **ExoGCM emulator:** {see field coregionalized decoder section of GPLVR.md}
- **observation noise $\sigma^2$** 
	- we use homoskedastic Gaussian noise $E_{ij}\sim \mathcal{N}(0, \sigma^2)$ with prior: $\sigma \sim \text{HalfNormal}(0.5^2)$ 

### Output reweighting $\boldsymbol{\alpha}$ 
- optional. not included in standard gplfr.