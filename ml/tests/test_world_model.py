

def test_linear_skip_starts_identical_and_can_learn_a_linear_map():
    """Zero-initialised skip: the model is the plain one at init; after a
    few steps on a purely linear two-lag process, mu tracks A[s_t, s_{t-1}]."""
    import torch
    from nidra.models.world_model import WorldModel
    torch.manual_seed(0)
    F, L, K = 6, 5, 2
    plain = WorldModel(n_features=F, hidden_size=8, encoder_layers=1, transition_mlp_hidden=16, linear_skip=False)
    skip = WorldModel(n_features=F, hidden_size=8, encoder_layers=1, transition_mlp_hidden=16, linear_skip=True)
    skip.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(3, L, F)
    with torch.no_grad():
        a = plain.rollout(x, K=K, stochastic=False).states
        b = skip.rollout(x, K=K, stochastic=False).states
    assert torch.allclose(a, b)
    assert skip.transition.skip.weight.abs().sum() == 0
    # train the skip only on s_{t+1} - s_t = 0.5 * s_{t-1}
    A = torch.zeros(F, 2 * F)
    A[:, F:] = 0.5 * torch.eye(F)
    opt = torch.optim.Adam(skip.transition.skip.parameters(), lr=5e-2)
    for _ in range(200):
        h = torch.randn(64, 8)
        s_t, s_prev = torch.randn(64, F), torch.randn(64, F)
        mu, _ = skip.transition(h, s_t, s_prev)
        target = torch.cat([s_t, s_prev], dim=-1) @ A.T
        mlp_mu, _ = plain.transition(h)
        loss = ((mu - mlp_mu.detach() - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.allclose(skip.transition.skip.weight, A, atol=0.05)
