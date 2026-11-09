"""Full natural-trajectory likelihood, frozen action-independent cost baseline."""
import torch


def fixed_reference_objective(log_sums, costs, reference_cost, weights=(1., .25, .25)):
    if len(log_sums) != 3 or len(costs) != 3 or tuple(weights) != (1., .25, .25):
        raise ValueError('Exactly the frozen three weighted trajectories required')
    z = log_sums[0]
    c = torch.as_tensor(costs, dtype=torch.float64, device=z.device).detach()
    baseline = torch.as_tensor(reference_cost, dtype=torch.float64, device=z.device).detach()
    if not bool(torch.isfinite(c).all() and torch.isfinite(baseline)):
        raise FloatingPointError('Nonfinite costs/reference')
    # A_return = -J_sample - (-J_reference). Never clamp or drop negative values.
    advantages = (baseline - c).detach()
    loss = sum(-w*a*lp for w, a, lp in zip(weights, advantages, log_sums))/sum(weights)
    return loss, advantages
