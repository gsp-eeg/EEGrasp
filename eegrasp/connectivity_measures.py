import numpy as np

def compute_phase_diff(s1, s2):
    """ 
    Compute the phase differences between two signals.

    Parameters
    ----------
    s1 : array_like
        The first signal.
    s2 : array_like
        The second signal.

    Returns
    -------
    phase_diff : array_like
        The phase differences between the two signals
    """
    from scipy.signal import hilbert
    # Compute the analytic signals
    analytic_s1 = hilbert(s1)
    analytic_s2 = hilbert(s2)
    
    # Compute the instantaneous phases
    phase1 = np.angle(analytic_s1)
    phase2 = np.angle(analytic_s2)

    # Compute the phase differences
    phase_diff = phase1 - phase2
    
    return phase_diff

def phase_coherence(s1, s2):
    """ 
    Compute the phase coherence between two signals.

    Parameters
    ----------
    s1 : array_like
        The first signal.
    s2 : array_like
        The second signal.

    Returns
    -------
    pc : float
        The phase coherence between the two signals
    """
    # Compute the phase differences
    phase_diff = compute_phase_diff(s1, s2)
    
    # Compute the phase coherence
    pc = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return pc

def imaginary_coherence(s1, s2):
    """ 
    Compute the imaginary part of coherence between two signals.

    Parameters
    ----------
    s1 : array_like
        The first signal.
    s2 : array_like
        The second signal.

    Returns
    -------
    ic : float
        The imaginary part of coherence between the two signals
    """
    # Compute the phase differences
    phase_diff = compute_phase_diff(s1, s2)
    
    # Compute the imaginary part of coherence
    ic = np.abs(np.mean(np.imag(np.exp(1j * phase_diff))))
    
    return ic

def phase_lag_index(s1, s2):
    """
    Compute the phase lag index between two signals.

    Parameters
    ----------
    s1 : array_like
        The first signal.
    s2 : array_like
        The second signal.

    Returns
    -------
    pli : float
        The phase lag index between the two signals
    """
    # Compute the phase differences
    phase_diff = compute_phase_diff(s1, s2)
    
    # Compute the phase lag index
    pli = np.abs(np.mean(np.sign(phase_diff)))
    
    return pli

def phase_locking_value(s1, s2):
    """
    Compute the phase locking value between two signals.

    Parameters
    ----------
    s1 : array_like
        The first signal.
    s2 : array_like
        The second signal.

    Returns
    -------
    plv : float
        The phase locking value between the two signals
    """
    # Compute the phase differences
    phase_diff = compute_phase_diff(s1, s2)
    
    # Compute the phase locking value
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv

if __name__ == "__main__":
    # Generate example signals (for testing purposes)
    np.random.seed(0)
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 10 * t) #+ 0.5 * np.random.randn(len(t))
    s2 = np.sin(2 * np.pi * 10 * t + np.pi/4 * t) #+ 0.5 * np.random.randn(len(t))

    # Compute the metrics
    pc = phase_coherence(s1, s2)
    ic = imaginary_coherence(s1, s2)
    pli = phase_lag_index(s1, s2)
    plv = phase_locking_value(s1, s2)

    print(f"Phase Coherence: {pc}")
    print(f"Imaginary Part of Coherence: {ic}")
    print(f"Phase Lag Index: {pli}")
    print(f"Phase Locking Value: {plv}")
