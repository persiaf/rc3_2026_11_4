"""GAE uses true task termination separately from a rollout collection cut."""
import torch


def compute_gae(rewards, values, next_values, terminated, gamma=1., lam=.95, trace_mask=None):
    """Return advantages, returns for T or T×B tensors.

    next_values refers to the actual next state, never an auto-reset state.
    Collection boundaries bootstrap; the backward trace naturally ends at the
    supplied T boundary. Optional trace_mask can cut interior collector segments
    without removing the next-state bootstrap. True termination cuts both.
    """
    if not 0<=gamma<=1 or not 0<=lam<=1:raise ValueError('invalid GAE coefficients')
    values=torch.as_tensor(values)
    if not values.dtype.is_floating_point:values=values.float()
    rewards=torch.as_tensor(rewards,dtype=values.dtype,device=values.device)
    next_values=torch.as_tensor(next_values,dtype=values.dtype,device=values.device)
    raw_terminated=torch.as_tensor(terminated,device=values.device)
    if raw_terminated.dtype != torch.bool:raise ValueError('terminated must be a Boolean tensor')
    terminated=raw_terminated
    if values.ndim not in (1,2) or not len(values) or any(x.shape!=values.shape for x in (rewards,next_values,terminated)):
        raise ValueError('GAE inputs must have identical nonempty T or T×B shapes')
    if any(not torch.isfinite(x).all() for x in (values,rewards,next_values)):
        raise ValueError('GAE inputs must be finite')
    continuation=(~terminated).to(values.dtype)
    if trace_mask is None:trace=continuation
    else:
        trace=torch.as_tensor(trace_mask,dtype=values.dtype,device=values.device)
        if trace.shape!=values.shape or not torch.isfinite(trace).all() or ((trace<0)|(trace>1)).any():
            raise ValueError('invalid trace mask')
        trace=trace*continuation
    advantages=torch.zeros_like(values);last=torch.zeros_like(values[-1])
    for index in reversed(range(len(values))):
        delta=rewards[index]+gamma*next_values[index]*continuation[index]-values[index]
        last=delta+gamma*lam*trace[index]*last
        advantages[index]=last
    return advantages,advantages+values
