import numpy as np
from Env.cell import get_mean_disturbance

def corporation_reward(corp, prev_capital, prev_disturbance, list_cell):
    """
    Compute reward for a corporation agent.
    Initial version only considers capital changes and disturbance changes：
    R = ΔCapital - ΔDisturbance

    Parameters
    ----------
    corp : Corporation
        current corporation agent object（has at least .capital, .biodiversity_score, .resilience attributes）
    prev_capital : float
        previous steps资Thislevel
    prev_disturbance : float
        previous stepsall cells  anthropogenic disturbance totaland

    Returns
    -------
    reward : float
    """
    mean_dist = get_mean_disturbance(list_cell)
    delta_capital = corp.capital - prev_capital
    delta_disturbance = mean_dist - prev_disturbance  # 假设 corp 有个 total_disturbance attributes
    return delta_capital - delta_disturbance


def investor_reward(investor, corporations, prev_portfolio_value):
    """
    Compute reward for an investor agent.
    初版只考虑portfolio valuechange：
    R = ΔPortfolioValue

    Parameters
    ----------
    investor : Investor
        current investor agent object（has at least .cash_level, .portfolio attributes）
    corporations : list of Corporation
        currentall corporation object
    prev_portfolio_value : float
        previous stepsportfolio value

    Returns
    -------
    reward : float
    """
    ### TODO: How to calculate the portfolio value needs to be discussed
    # Calculateportfolio value
    portfolio_value = 0.0
    for i, corp in enumerate(corporations):
        portfolio_value += investor.portfolio[i] * corp.capital

    delta_value = portfolio_value - prev_portfolio_value
    return delta_value


def compute_all_rewards(corporations, investors, prev_states, list_cell):
    """
    统一Calculateall agents  reward

    Parameters
    ----------
    corporations : list of Corporation
    investors : list of Investor
    prev_states : dict
        previous stepsstate快照，包括：
        {
            "corporations": [ (capital, disturbance), ...],
            "investors": [ portfolio_value, ...]
        }

    Returns
    -------
    rewards : dict
        {
            "corporations": [r1, r2, ...],
            "investors": [r1, r2, ...]
        }
    """
    corp_rewards = []
    for corp, (prev_cap, prev_dist) in zip(corporations, prev_states["corporations"]):
        corp_rewards.append(corporation_reward(corp, prev_cap, prev_dist, list_cell))

    inv_rewards = []
    for inv, prev_val in zip(investors, prev_states["investors"]):
        inv_rewards.append(investor_reward(inv, corporations, prev_val))

    return {"corporations": corp_rewards, "investors": inv_rewards}
