2026-01-29
Type: #idea 
Topics: 
References: 

---
# Toy for GPLFR

- The advantage of GPLFR over PCA+MOGP is strongest when outputs $\mathbf{y}$ have significant variation that is independent of the inputs $\mathbf{x}$ and *structured* (i.e., not isotropic noise). 
- With a tight latent budget, PCA will spend components on this unpredictable 'nuisance' subspace; GPLFR will preferentially allocate capacity to the predictable 'signal' subspace because the latents are constrained by a GP over $\mathbf{x}$.
- We demonstrate this first for a toy problem where we add unpredictability via nuisance latents that are i.i.d. across samples. 
## Data-Generating Process (forward model)
- We generate outputs $\mathbf{y} \in \mathbb{R}^{D_y}$ from inputs $\mathbf{x} \in \mathbb{R}^{D_x}$ as $$\mathbf{y}=\mathbf{W}_\text{sig} \mathbf{z}_\text{sig}(\mathbf{x})+ \mathbf{W}_\text{nuis} \mathbf{z}_\text{nuis}+ \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon}\sim \mathcal{N}(0, \sigma_\epsilon^2 \mathbf{I}_{D_y}).$$
- **predictable variation**
	- choose $D_\text{sig}$ signal latents and draw each as a function of $\mathbf{x}$: $$z_\text{sig}^{(q)}\sim \mathcal{GP}(0,\,\sigma_\text{sig}^2 k)\quad \text{for } q=1,...,D_\text{sig}.$$
	- for training inputs $\{\mathbf{x}_i\}_{i=1}^N$, this implies $$\mathbf{z}^{(q)}_\text{sig}\equiv\big(z_\text{sig}^{(q)}(\mathbf{x}_1),...,z_\text{sig}^{(q)}(\mathbf{x}_N)\big)^\top \sim \mathcal{N}(0,\, \sigma_\text{sig}^2 \mathbf{K}), \quad K_{ij}=k(\mathbf{x}_i, \mathbf{x}_j).$$
	- stacking across $q$: $\mathbf{Z}_\text{sig}\in \mathbb{R}^{N\times D_\text{sig}}$ where the $q$-th column is $\mathbf{z}^{(q)}_\text{sig}$.
- **nuisance variation**
	- nuisance latents are i.i.d. across samples: $$\mathbf{z}_\text{nuis}\sim \mathcal{N}(0, \sigma^2_\text{nuis}\mathbf{I}_{D_\text{nuis}}).$$ Stacking gives $\mathbf{Z}_\text{nuis}\in \mathbb{R}^{N\times D_\text{nuis}}$.
- **output structure:**
    - Outputs live on a 2D grid with $D_y = H \times W$ and coordinates $(u, v)$ where $u =0, \ldots, H-1$, $v =0, \ldots, W-1$.
    - Columns of $\mathbf{W}_\text{sig}$ are localized Gaussian blobs with centres $(\bar{u}_p, \bar{v}_p)$ and scales $s_p$: $$\phi_\text{sig}^{(p)}(u,v) = \exp\!\Big({-}\frac{(u - \bar{u}_p)^2 + (v - \bar{v}_p)^2}{2s_p^2}\Big).$$for $p=1,...,D_\text{sig}$
    - Columns of $\mathbf{W}_\text{nuis}$ are low-frequency 2D DCT modes with indices $(k_r, \ell_r)$: $$\phi_\text{nuis}^{(r)}(u,v) = \cos\!\Big(\frac{\pi k_r (u + \tfrac{1}{2})}{H}\Big) \cos\!\Big(\frac{\pi \ell_r (v + \tfrac{1}{2})}{W}\Big).$$for $r=1,...,D_\text{nuis}$ 
    - We stack all basis functions and orthonormalize jointly via QR decomposition, using nuisance-first ordering so $\mathbf{W}_\text{nuis}$ remains low-frequency/global after orthonormalization (and $\mathbf{W}_\text{sig}$ becomes blob-like but orthogonal to nuisance).
	- :obs_note_glyph:code
		- $\bar{u}_p, \bar{v}_p$ are sampled uniformly over grid
		- $s_p$ is sampled uniformly from $[0.8,2.2]$
- **Inputs** are drawn as $\mathbf{x}_i \sim \mathcal{N}(0, \mathbf{I}_{D_x})$ and split randomly into train/val/test.
- The ratio $\sigma_\text{nuis}/\sigma_\text{sig}$ controls the relative influence of unpredictable variance.
	- :obs_note_glyph: to get the clean “PCA spends first $D_\text{nuis}$​ PCs on nuisance” regime, we want $\sigma^2_\text{nuis}>\lambda_\max(\text{Cov}(\mathbf{Z}_\text{sig}))\approx \sigma^2_\text{sig}$ (approximate due finite sample size). We set $\sigma_\text{nuis}/\sigma_\text{sig}=3$ so this holds robustly.
- **settings**
	- RBF kernel with lengthscale $\ell=1$ shared across input dimensions

## Evaluation
- The true conditional mean is $\mathbf{y}_\text{sig} \equiv \mathbb{E}[\mathbf{y} \mid \mathbf{x}] = \mathbf{W}_\text{sig},\mathbf{z}_\text{sig}(\mathbf{x})$, since nuisance and noise terms have zero mean.
- **Conditional mean error:** $$\text{RMSE}_\text{sig} = \sqrt{\frac{1}{N_\text{test} D_y} \sum_i |\hat{\mathbf{y}}_i - \mathbf{y}_{\text{sig},i}|_2^2}.$$
- **Observed error:** $$\text{RMSE}_\text{obs} = \sqrt{\frac{1}{N_\text{test} D_y} \sum_i |\hat{\mathbf{y}}_i - \mathbf{y}_i|_2^2}.$$
- **Per-PC $R^2$** (PCA+GP baseline): for principal component $d$, let $s_{i,d}$ be the score from projecting observed $\mathbf{y}_i$ onto the $d$-th PC, and $\hat{s}_{i,d}$ the GP prediction from $\mathbf{x}_i$. Then $$R_d^2 = 1 - \frac{\sum_i (\hat{s}_{i,d} - s_{i,d})^2}{\sum_i (s_{i,d} - \bar{s}_d)^2}.$$

## Empirical notes (2026-01-30)
- **Scale:** On `data/data.npz`, $\mathrm{std}(Y)\approx 0.098$ and $\mathrm{std}(Y_\text{sig})\approx 0.029$; any $\mathrm{RMSE}_\text{sig}\approx 0.029$ is basically “predict zero”.
- **PCA+GP:** With `sigma_xi2_mode=shared` (learned), even with `n_components >= D_sig + D_nuis`, the shared nugget often grows to accommodate nuisance-dominated PCs and hurts the predictable PCs (RMSE_sig stays ~0.03). Allowing `sigma_xi2_mode=per-latent` or fixing `sigma_xi2_mode=float` with a tiny value (e.g. `1e-5`) recovers the signal (RMSE_sig ~0.02 at `n_train=200`).
- **GPLFR (MAP):** On this dataset, GPLFR tends to allocate latent capacity to nuisance structure and/or inflate `sigma_xi2`, yielding predictions close to the mean (RMSE_sig ~0.03). The best init we found uses `pca_init.whiten_scores=false`; learning `sigma_f` (`sigma_f_mode=per_latent`) was unstable here.
