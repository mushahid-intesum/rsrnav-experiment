import numpy as np


def success_rate(episode_infos):
    """Compute Success Rate from a list of episode info dicts."""
    if not episode_infos:
        return 0.0
    return np.mean([float(info.get("success", False)) for info in episode_infos])


def spl(episode_infos):
    """
    Compute Success weighted by Path Length (Anderson et al., 2018).

    SPL = (1/N) Σ S_i * (l_i / max(p_i, l_i))

    where S_i = success, l_i = shortest path, p_i = actual path length.
    """
    if not episode_infos:
        return 0.0

    values = []
    for info in episode_infos:
        s = float(info.get("success", False))
        shortest = info.get("episode_distance", 1.0)
        actual = info.get("num_steps", 1) * 0.25  # approx path from step count
        if shortest <= 0:
            shortest = 1e-5
        values.append(s * shortest / max(actual, shortest))

    return np.mean(values)


def soft_spl(episode_infos):
    """
    Soft SPL — continuous version that doesn't require binary success.
    Uses (1 - d_final / d_start) as a soft success measure.
    """
    if not episode_infos:
        return 0.0

    values = []
    for info in episode_infos:
        d_start = info.get("episode_distance", 1.0)
        d_final = info.get("distance_to_goal", d_start)
        if d_start <= 0:
            d_start = 1e-5

        soft_success = max(0, 1.0 - d_final / d_start)
        shortest = d_start
        actual = info.get("num_steps", 1) * 0.25
        values.append(soft_success * shortest / max(actual, shortest))

    return np.mean(values)


def distance_to_success(episode_infos):
    """Mean final distance to goal across episodes."""
    if not episode_infos:
        return float("inf")
    return np.mean([info.get("distance_to_goal", float("inf"))
                     for info in episode_infos])


def compute_all_metrics(episode_infos):
    """Compute all navigation metrics and return as a dict."""
    return {
        "SR": success_rate(episode_infos),
        "SPL": spl(episode_infos),
        "SoftSPL": soft_spl(episode_infos),
        "DTS": distance_to_success(episode_infos),
        "num_episodes": len(episode_infos),
    }


def format_metrics(metrics):
    """Format metrics dict as a readable string."""
    parts = [
        f"SR: {metrics['SR']:.3f}",
        f"SPL: {metrics['SPL']:.3f}",
        f"SoftSPL: {metrics['SoftSPL']:.3f}",
        f"DTS: {metrics['DTS']:.2f}m",
        f"N={metrics['num_episodes']}",
    ]
    return " | ".join(parts)
