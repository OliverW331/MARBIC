# Agents/investor.py
import numpy as np
from typing import List, Any

class Investor:
    """
    Simple Investor class：
    - cash: cash
    - portfolio: length of Nc  vector，represents funds proportionally allocated to each corporation（sum <= 1）
    """

    def __init__(self, investor_id: int, cash: float = 100.0, n_corporations: int = 1):
        self.id = int(investor_id)
        self.cash = float(cash)
        self.n_corp = int(n_corporations)
        self.portfolio = np.zeros(self.n_corp, dtype=float)  # proportions, sum <= 1

    def reset(self):
        self.cash = 100.0
        self.portfolio = np.zeros(self.n_corp, dtype=float)

    def invest(self, weights: List[float], corporations: List[Any], normalize: bool = True):
        """
        weights: length Nc non-negative array（environment will ensure sum(weights) <= 1，or这里归一化）
        corporations: Corporation 实例list，invest 会把cash转为对 corp 资This注入（简单建模）
        return字典式executeinformation
        """
        ### TODO： 考虑can以用神经网络andsoftmax来生成weights，决定投资比例
        ### 暂时userandom生成weight
        w = np.array(weights, dtype=float).reshape(self.n_corp)
        # clip negatives and normalize if needed
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if normalize and s > 1.0:
            w = w / s  # 使totaland =1（environmentfinal还会乘以 cash）
            s = w.sum()

        invest_fraction = s  # 比例
        invested_amount = self.cash * invest_fraction
        # 从cash扣除
        self.cash -= invested_amount

        if invested_amount > 0 and s > 0:
            # allocationto各 corporation（按 w 相comparison例）
            for i, corp in enumerate(corporations):
                alloc = invested_amount * (w[i] / s) if s > 0 else 0.0
                corp.capital += alloc

        self.portfolio = w.copy()
        return {"invested_amount": invested_amount, "weights": w.copy()}

    def to_dict(self):
        return {"id": self.id, "cash": self.cash, "portfolio": self.portfolio.tolist()}
