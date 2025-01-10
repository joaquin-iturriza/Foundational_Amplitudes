import numpy as np


def preprocess_particles(particles_raw, trafos=None, mean=None, std=None, eps_std=1e-2):
    assert np.isfinite(particles_raw).all()
    
    if trafos:
    
    else:
        features = particles_raw
    
    if mean is None or std is None:
        mean = features.mean((0, 1))
        std = features.std((0, 1))
        std = np.clip(std, a_min=eps_std, a_max=None)  # avoid std=0.

    features = (features - mean) / std
    assert np.isfinite(features).all()
    return features, mean, std


def preprocess_amplitude(amplitude, std=None):
    log_amplitude = np.log(amplitude)
    if std is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    assert np.isfinite(prepd_amplitude).all()
    return prepd_amplitude, mean, std


def undo_preprocess_amplitude(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    log_amplitude = prepd_amplitude * std + mean
    amplitude = np.exp(log_amplitude)
    return amplitude
