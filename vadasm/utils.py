import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def ties_merge(deltas: torch.Tensor, drop_rate: float = 0.3, rescale_factor: float = 1.0/0.7):
    """
    TIES (Trim, Elect, Sign and Merge) merging algorithm.
    
    Args:
        deltas: Tensor of shape (num_models, ...) containing parameter deltas
        drop_rate: Fraction of deltas to drop
        rescale_factor: Factor to rescale survivors
    
    Returns:
        Merged delta tensor
    """
    # Sign conflicts: where different models have different signs
    signs = torch.sign(deltas)
    mean_sign = torch.mean(signs.float(), dim=0)
    
    # Keep deltas where signs mostly agree or magnitude is large
    sign_agreement = (torch.abs(mean_sign) > 0.5).float()
    magnitude_threshold = torch.quantile(torch.abs(deltas), drop_rate, dim=0)
    magnitude_mask = (torch.abs(deltas) > magnitude_threshold).any(dim=0)
    
    # Combine masks: keep if either agreement or large magnitude
    keep_mask = sign_agreement | magnitude_mask
    
    # Apply TIES: keep largest magnitude when signs conflict
    abs_deltas = torch.abs(deltas)
    max_abs_values, max_indices = torch.max(abs_deltas, dim=0)
    
    final_delta = torch.zeros_like(deltas[0])
    for i in range(len(final_delta.view(-1))):
        flat_idx = i
        idx = np.unravel_index(i, final_delta.shape)
        if keep_mask.view(-1)[flat_idx]:
            winner_idx = max_indices.view(-1)[flat_idx]
            final_delta.view(-1)[flat_idx] = deltas[winner_idx].view(-1)[flat_idx]
        else:
            final_delta.view(-1)[flat_idx] = 0
    
    return final_delta * rescale_factor

def dare_merge(deltas: torch.Tensor, drop_rate: float = 0.3):
    """
    DARE (Drop And REscale) sparsification.
    
    Args:
        deltas: Parameter deltas to sparsify 
        drop_rate: Fraction to drop (keep (1-drop_rate) survivors)
    
    Returns:
        Sparsified deltas
    """
    # Drop smallest magnitude deltas
    abs_deltas = torch.abs(deltas)
    threshold = torch.quantile(abs_deltas, drop_rate)
    keep_mask = abs_deltas > threshold
    
    # Rescale survivors
    num_kept = keep_mask.sum()
    rescale_factor = deltas.numel() / num_kept if num_kept > 0 else 1.0
    
    return deltas * keep_mask.float() * rescale_factor

def hungarian_neuron_alignment(model_a_params: torch.Tensor, model_b_params: torch.Tensor):
    """
    Align neurons using Hungarian algorithm based on cosine similarity.
    
    Args:
        model_a_params: Parameters from model A (neurons x features)
        model_b_params: Parameters from model B (neurons x features)
    
    Returns:
        Permutation tensor mapping model B neurons to model A
    """
    # Cosine similarity matrix
    cos_sim = cosine_similarity(
        model_a_params.cpu().numpy(),
        model_b_params.cpu().numpy()
    )
    
    # Hungarian algorithm gives optimal assignment
    # We use negative because linear_sum_assignment minimizes
    row_ind, col_ind = linear_sum_assignment(-cos_sim)
    
    # Create permutation tensor
    perm = torch.zeros(model_b_params.shape[0], dtype=torch.long)
    perm[row_ind] = col_ind
    
    return perm

def svd_subspace_reduction(weight_matrix: torch.Tensor, variance_threshold: float = 0.95):
    """
    Reduce parameter subspace using SVD.
    
    Args:
        weight_matrix: Weight matrix to reduce
        variance_threshold: Fraction of variance to preserve
    
    Returns:
        Reduced weight matrix
    """
    # SVD decomposition
    U, s, Vt = torch.svd(weight_matrix)
    
    # Keep components capturing threshold variance
    explained_variance_ratio = torch.cumsum(s**2, 0) / torch.sum(s**2)
    num_components = (explained_variance_ratio < variance_threshold).sum() + 1
    
    # Reduce to subspace
    U_reduced = U[:, :num_components]  
    s_reduced = s[:num_components]
    Vt_reduced = Vt[:num_components, :]
    
    return U_reduced @ torch.diag(s_reduced) @ Vt_reduced

def deap_hyperparameter_optimization(objective_func, param_bounds, pop_size=30, generations=15):
    """
    DEAP-based evolutionary optimization for hyperparameters.
    
    Args:
        objective_func: Function to minimize (takes param dict, returns float)
        param_bounds: Dict of {'param_name': (min_val, max_val)}
        pop_size: Population size
        generations: Number of generations
    
    Returns:
        Best parameter configuration
    """
    try:
        from deap import base, creator, tools, algorithms
        import random
    except ImportError:
        raise ImportError("DEAP library required for evolutionary optimization")
    
    # Define fitness (minimization)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Register parameter generators
    for param_name, (min_val, max_val) in param_bounds.items():
        toolbox.register(f"attr_{param_name}", random.uniform, min_val, max_val)
    
    # Individual is list of parameters in bounds order  
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [getattr(toolbox, f"attr_{name}") for name in param_bounds.keys()])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_individual(individual):
        param_dict = dict(zip(param_bounds.keys(), individual))
        return objective_func(param_dict),
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations, 
                       stats=stats, halloffame=hof, verbose=False)
    
    best_params = dict(zip(param_bounds.keys(), hof[0]))
    return best_params