from .registry import REGISTRY


def get_latency_ms(expert_id: str) -> float:
    return float(REGISTRY.get(expert_id, {}).get('latency_ms', 0.0))


def get_cost_per_1k(expert_id: str) -> float:
    return float(REGISTRY.get(expert_id, {}).get('cost_usd_per_1k_tokens', 0.0))

def get_param_count(expert_id: str) -> int:
    return int(REGISTRY.get(expert_id, {}).get('n_params', 0))

def compute_cost(expert_id: str, n_tokens: int, cost_type="usd") -> float:
    if cost_type == "latency":
        return get_latency_ms(expert_id) * n_tokens
    elif cost_type == "n_params":
        return get_param_count(expert_id) * 0.03
    elif cost_type == "usd":
        return get_cost_per_1k(expert_id) * 200
    
    raise ValueError(f"Unknown cost type: {cost_type}. Use 'latency', 'n_params', or 'usd'.")