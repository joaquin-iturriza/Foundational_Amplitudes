import numpy as np
import torch
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer


def preprocess_amplitude(amplitude, trafos=None):
    mean, std = 0, 0
    if trafos:
        for fn_str in trafos:
            if (
                fn_str == "standardization"
            ):  # treat "standardize" separately because we need to save stds and means
                amplitude, mean, std = standardization(
                    amplitude, return_mean_std=True, clip=False
                )
            else:
                fn = get_fn(fn_str)
                amplitude = fn(amplitude, None)
    assert np.isfinite(amplitude).all()
    return amplitude, mean, std


def undo_preprocess_amplitude(amplitude, mean, std, trafos=None):
    if trafos:
        trafos = trafos[::-1] # reverse order of trafos
        for fn_str in trafos:
            if fn_str == "standardization":
                amplitude = amplitude * std + mean
            else:
                inv_fn = get_inv_fn(fn_str)
                amplitude = np.minimum(30, inv_fn(amplitude, None))

            assert np.isfinite(amplitude).all(), f'{fn_str} failed'

    return amplitude


def preprocess_particles(
    particles_raw,
    type_tokens,
    trafos=None,
    incl_fvs=True,
    mean=None,
    std=None,
    eps_std=1e-2,
    return_dict=False,
):
    assert np.isfinite(particles_raw).all()

    feature_sets = {}
    if trafos:
        for t_name, t_fns in trafos.items():
            if not t_fns:  # skip empty transformation lists
                continue
            transformed_features = particles_raw
            for fn_str in t_fns:
                print(fn_str)
                fn = get_fn(fn_str)
                transformed_features = fn(transformed_features, type_tokens)
            feature_sets[t_name] = transformed_features

    if incl_fvs:
        feature_sets["fvs_raw"] = sort_particles(particles_raw, type_tokens, "E")
        feature_sets["fvs_raw"] /= np.max(
            np.abs(feature_sets["fvs_raw"][:, ::4])
        )  # normalize by max energy of particles
        # feature_sets["fvs_raw"] = particles_raw
        print(feature_sets["fvs_raw"].shape)

    if return_dict:
        return feature_sets
    else:
        return np.concatenate(list(feature_sets.values()), axis=1)


def standardization(features, return_mean_std=False, clip=True):
    """standardize features"""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    if clip:
        std = np.clip(std, a_min=1e-6, a_max=None)  # avoid std=0.
    features = (features - mean) / std
    assert np.isfinite(features).all()
    if return_mean_std:
        return features, mean, std
    else:
        return features


def compute_invariants(particles, eps=1e-4, incl_diag_invariants=False, reshape=True):
    """compute Lorentz invariants out of four-vectors"""
    print('#####################')
    print(particles.shape)
    print('#####################')
    if reshape:
        ps = particles.reshape(particles.shape[0], particles.shape[1] // 4, 4)
    else:
        ps = particles

    # compute matrix of all inner products
    def inner_product(p1, p2):
        return p1[..., 0] * p2[..., 0] - (p1[..., 1:] * p2[..., 1:]).sum(axis=-1)

    if incl_diag_invariants:
        offset = 0
    else:
        offset = 1

    idxs = np.triu_indices(ps.shape[-2], k=offset)
    invariants = inner_product(ps[..., idxs[0], :], ps[..., idxs[1], :])
    if type(invariants) == np.ndarray:
        invariants = np.clip(invariants, a_min=eps, a_max=None)
    else:
        invariants = torch.clip(invariants, eps, None)
    return invariants


def sort_particles(particles, type_tokens, sort_key):
    """sort particles according to some property"""
    ps = particles.reshape(particles.shape[0], particles.shape[1] // 4, 4)
    particle_types = np.unique(type_tokens)
    sorted_ps = []
    for particle_type in particle_types:
        mask = type_tokens == particle_type
        # sort according to energy, which is the first entry of the four-vector
        match sort_key:
            case "E":
                sorted_indices = np.argsort(ps[:, mask][:, :, 0])
            case "pt":
                pts = np.linalg.norm(ps[:, mask][:, :, 1:3], axis=-1)
                sorted_indices = np.argsort(pts)
            case _:
                raise ValueError(f"Unknown sort key {sort_key}")
        sorted_ps.append(
            np.take_along_axis(ps[:, mask], sorted_indices[:, :, np.newaxis], axis=1)
        )
        #print(sorted_ps)
    return np.concatenate(sorted_ps, axis=1).reshape(particles.shape)


def apply_boxcox(particles):
    ps = particles.transpose()
    res = np.zeros(ps.shape)
    for i in range(ps.shape[0]):
        try:
            res[i] = boxcox(ps[i])[0]
        except:
            res[i] = ps[i]
    return res.transpose()


def apply_quantile_transform(particles):
    return QuantileTransformer(output_distribution="normal").fit_transform(particles)


def get_fn(fn_str):
    # get functions from string
    match fn_str:
        case "None":
            return lambda p, t: p
        case "log":
            return lambda p, t: np.log(p)
        case "exp":
            return lambda p, t: np.exp(p)
        case "sqrt":
            return lambda p, t: np.sqrt(p)
        case "inverse":
            return lambda p, t: 1 / p
        case "boxcox":
            return lambda p, t: apply_boxcox(p)
        case "quantile_transform":
            return lambda p, t: apply_quantile_transform(p)
        case "invs":
            return lambda p, t: compute_invariants(p)
        case "standardization":
            return lambda p, t: standardization(p)
        case "sort_E":
            return lambda p, t: sort_particles(p, t, "E")
        case "sort_pt":
            return lambda p, t: sort_particles(p, t, "pt")
        case _:
            raise ValueError(f"Unknown transformation function {fn_str}")


def get_inv_fn(fn_str):
    # get inverse functions from string
    match fn_str:
        case "log":
            return lambda p, t: np.exp(p)
        case "exp":
            return lambda p, t: np.log(p)
        case "sqrt":
            return lambda p, t: p**2
        case "inverse":
            return lambda p, t: 1 / p
        case _:
            raise ValueError(f"Unknown inverse transformation function {fn_str}")
