import numpy as np
from Env.cell import get_mean_disturbance

def corporation_reward(corp, prev_capital, prev_disturbance, list_cell):
    """
    Compute reward for a corporation agent.
    初版只考虑资本变化和干扰变化：
    R = ΔCapital - ΔDisturbance

    Parameters
    ----------
    corp : Corporation
        当前 corporation agent 对象（至少有 .capital, .biodiversity_score, .resilience 属性）
    prev_capital : float
        上一步的资本水平
    prev_disturbance : float
        上一步所有 cells 的 anthropogenic disturbance 总和

    Returns
    -------
    reward : float
    """
    mean_dist = get_mean_disturbance(list_cell)
    delta_capital = corp.capital - prev_capital
    delta_disturbance = mean_dist - prev_disturbance  # 假设 corp 有个 total_disturbance 属性
    return delta_capital - delta_disturbance


def investor_reward(investor, corporations, prev_portfolio_value):
    """
    Compute reward for an investor agent.
    初版只考虑投资组合价值变化：
    R = ΔPortfolioValue

    Parameters
    ----------
    investor : Investor
        当前 investor agent 对象（至少有 .cash_level, .portfolio 属性）
    corporations : list of Corporation
        当前所有 corporation 对象
    prev_portfolio_value : float
        上一步的投资组合价值

    Returns
    -------
    reward : float
    """
    ### TODO: How to calculate the portfolio value needs to be discussed
    # 计算投资组合价值
    portfolio_value = 0.0
    for i, corp in enumerate(corporations):
        portfolio_value += investor.portfolio[i] * corp.capital

    delta_value = portfolio_value - prev_portfolio_value
    return delta_value


def compute_all_rewards(corporations, investors, prev_states, list_cell):
    """
    统一计算所有 agents 的 reward

    Parameters
    ----------
    corporations : list of Corporation
    investors : list of Investor
    prev_states : dict
        上一步的状态快照，包括：
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
