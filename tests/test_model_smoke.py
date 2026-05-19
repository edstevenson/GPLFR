from __future__ import annotations

import numpy as np

from gplfr import GPLFR, create_synthetic_data


def test_gplfr_model_fit_and_predict_smoke() -> None:
    data = create_synthetic_data(N=64, Dx=2, H=6, W=6, D_sig=3, sigma_nuis=0.3, sigma_eps=0.05, seed=0)
    X_train, Y_train = data["X"][:48], data["Y"][:48]
    X_test = data["X"][48:]

    model = GPLFR(latent_dim=3, kernel="rbf", amplitude_grouping="fixed", amplitude=1.0, inverse_temperature=0.3, device="cpu")
    fit_result = model.fit(X_train, Y_train, num_steps=60, verbose=False, seed=0)
    pred = model.predict(X_test)

    assert np.isfinite(fit_result.final_loss)
    assert pred.shape == (len(X_test), 36)
