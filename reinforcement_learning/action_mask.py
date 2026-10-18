"""Legal-action mask. Must stay consistent with SumoEnvironment.legalise_action."""
from __future__ import annotations

import numpy as np

from config.constants import NUM_PHASES


def legal_action_mask(env) -> np.ndarray:
    """Boolean mask over phases for the NEXT decision.

    * elapsed_green < min_green      -> only the active phase (hold) is legal
    * elapsed_green + interval > max -> holding is illegal, any other phase is legal
    * otherwise                      -> every phase is legal
    """
    sig, cfg = env.signal, env.cfg
    mask = np.ones(NUM_PHASES, dtype=bool)
    if sig.elapsed_green < cfg.min_green:
        mask[:] = False
        mask[sig.active_phase] = True
    elif sig.elapsed_green + cfg.decision_interval > cfg.max_green:
        mask[sig.active_phase] = False
    return mask