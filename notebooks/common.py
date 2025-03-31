

import itertools
import numpy as np
from evoscaper.utils.math import bin_array
from evoscaper.model.vae import sample_z
import jax
import umap
from sklearn.manifold import TSNE


def get_model_latent_space_dimred(p, rng, encoder, h2mu, h2logvar,
                                  x, cond, use_h,
                                  y_datanormaliser, y_methods_preprocessing,
                                  config_dataset, x_datanormaliser, x_methods_preprocessing,
                                  hpos_all,
                                  method='UMAP', n_show=1000, random_state=0,
                                  perplexity=30):
    
    rng_key = jax.random.PRNGKey(random_state)  # Initialize a PRNG key with a seed

    h_all = encoder(p, rng, np.concatenate([x, cond], axis=-1))
    h_all = h_all.reshape(np.prod(h_all.shape[:-1]), -1)
    mu = h2mu(p, rng, h_all)
    logvar = h2logvar(p, rng, h_all)
    z_all = sample_z(mu, logvar, rng, deterministic=True)
    if use_h:
        emb = h_all
    else:
        emb = z_all

    cond_rev_all = np.concatenate([y_datanormaliser.create_chain_preprocessor_inverse(y_methods_preprocessing)(
        cond[..., i], col=c).flatten() for i, c in enumerate(config_dataset.objective_col)]).reshape(np.prod(cond.shape[:-1]), -1).squeeze()
    # cond_rev_all = y_datanormaliser.create_chain_preprocessor_inverse(y_methods_preprocessing)(cond, col=config_dataset.objective_col[0]).reshape(np.prod(cond.shape[:-1]), -1).squeeze()
    x_rev_all = x_datanormaliser.create_chain_preprocessor_inverse(
        x_methods_preprocessing)(x).reshape(np.prod(x.shape[:-1]), -1).squeeze()

    x_bin_all, edges, labels = bin_array(x_rev_all, num_bins=10)
    x_bin_all = np.round(x_bin_all, 1)

    cond_binned = cond
    if not hpos_all['prep_y_categorical']:
        cond_binned = bin_array(
            cond_rev_all, num_bins=hpos_all['prep_y_categorical_n_bins'])[0]
        if cond.shape[-1] != cond_binned.shape[-1]:
            cond_binned = cond_binned.reshape(-1, cond.shape[-1])

    cond_unique = [np.unique(cond_binned[..., i])
                   for i in range(cond_binned.shape[-1])]
    cond_unique = np.array(list(itertools.product(*cond_unique)))

    idxs_show = []
    for c in cond_unique:
        idxs_show.extend(np.where((cond_binned != c).sum(axis=-1) == 0)
                         [0][:np.max([n_show//len(cond_unique), 5])])
    idxs_show = np.array(idxs_show)
    if len(idxs_show) > n_show:
        idxs_show = jax.random.choice(
            rng_key, idxs_show, (n_show,), replace=False)

    if method == 'UMAP':
        reducer = umap.UMAP(n_neighbors=100, n_components=2, random_state=random_state,
                            #  metric='euclidean', n_epochs=2000, learning_rate=0.1, init='spectral')
                            metric='euclidean', n_epochs=2000, learning_rate=0.1, init='pca')
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state)
    result_dimred = reducer.fit_transform(emb[idxs_show])

    return result_dimred, idxs_show, cond_unique, cond_binned, x_bin_all, x_rev_all, cond_rev_all, emb, h_all, z_all
