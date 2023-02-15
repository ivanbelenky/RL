from typing import (
    Tuple, 
    Sequence,  
    Any
)

import numpy as np
from numpy.linalg import norm as lnorm

from rl.model_free import (
    ModelFree,
    ModelFreePolicy,
    EpsilonSoftPolicy
)
from rl.utils import (
    Policy,
    _typecheck_all,
    _get_sample_step,
    _check_ranges,
    VQPi,
    Samples,
    Transition,
    Vpi,
    Qpi,
    PQueue,
    MAX_ITER,
    MAX_STEPS,
    TOL
)

