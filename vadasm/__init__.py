"""
V-ADASM initialization
"""

from .merger import VADASMMerger, ModelConfig, MergeConfig
from .vision import VisionExtractor
from .alignment import CrossModalAligner
from .fusion import SubspaceFuser
from .evolution import EvolutionaryTuner
from .utils import ties_merge, dare_merge, hungarian_neuron_alignment, svd_subspace_reduction

__version__ = "0.1.0"
__all__ = [
    "VADASMMerger", "ModelConfig", "MergeConfig",
    "VisionExtractor", "CrossModalAligner", 
    "SubspaceFuser", "EvolutionaryTuner",
    "ties_merge", "dare_merge", "hungarian_neuron_alignment", 
    "svd_subspace_reduction"
]