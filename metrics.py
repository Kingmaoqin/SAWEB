import torch

__all__ = ["cindex_fast"]

def cindex_fast(durations, events, risks):
    """
    Vectorized C-index. Larger risk -> more likely event earlier.
    durations: [N] float tensor
    events:    [N] 0/1 int/float tensor
    risks:     [N] float tensor (e.g., sum of hazards or cumulative hazard)
    """
    # durations = durations.view(-1)
    # events    = events.view(-1)
    # risks     = risks.view(-1)
    risks = torch.as_tensor(risks)
    device = torch.device(risks.device)
    durations = torch.as_tensor(durations, device=device).view(-1)
    events    = torch.as_tensor(events, device=device).view(-1)
    risks     = risks.to(device).view(-1)
    # Only comparable pairs where one is an event and the other is at risk longer
    # Create pairwise masks: i < j to avoid double counting
    N = durations.numel()
    idx = torch.arange(N, device=durations.device)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    mask_upper = (i < j)

    # Comparable if min(dur_i, dur_j) is the event time and that one has event=1
    d_i = durations.view(-1, 1)
    d_j = durations.view(1, -1)
    e_i = events.view(-1, 1)
    e_j = events.view(1, -1)

    comparable = torch.zeros_like(mask_upper, dtype=torch.bool)
    # i event before j censor/event
    comparable |= (e_i == 1) & (d_i < d_j)
    # j event before i censor/event
    comparable |= (e_j == 1) & (d_j < d_i)
    comparable = comparable & mask_upper

    if comparable.sum() == 0:
        return torch.tensor(0.5, device=durations.device)

    r_i = risks.view(-1, 1)
    r_j = risks.view(1, -1)

    # For pairs where i event earlier -> should have risk_i > risk_j
    # Define sign based on which duration is smaller
    sign = torch.sign(d_j - d_i)  # positive if i earlier
    # Prediction correct if (r_i - r_j) * sign > 0
    diff = (r_i - r_j) * sign
    concordant = (diff > 0) & comparable
    ties = (diff == 0) & comparable

    c = concordant.sum().float() + 0.5 * ties.sum().float()
    return c / comparable.sum().float()
