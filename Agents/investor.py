# Agents/investor.py
import numpy as np
from typing import List, Any

class Investor:
    """
    简单的 Investor 类：
    - cash: 现金
    - portfolio: 长度为 Nc 的向量，表示资金按比例分配到每个 corporation（sum <= 1）
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
        weights: length Nc 的非负数组（环境会确保 sum(weights) <= 1，或这里归一化）
        corporations: Corporation 实例列表，invest 会把现金转为对 corp 的资本注入（简单建模）
        返回字典式的执行信息
        """
        ### TODO： 考虑可以用神经网络和softmax来生成weights，决定投资比例
        ### 暂时使用随机生成的权重
        w = np.array(weights, dtype=float).reshape(self.n_corp)
        # clip negatives and normalize if needed
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if normalize and s > 1.0:
            w = w / s  # 使总和 =1（环境最终还会乘以 cash）
            s = w.sum()

        invest_fraction = s  # 比例
        invested_amount = self.cash * invest_fraction
        # 从现金扣除
        self.cash -= invested_amount

        if invested_amount > 0 and s > 0:
            # 分配到各 corporation（按 w 的相对比例）
            for i, corp in enumerate(corporations):
                alloc = invested_amount * (w[i] / s) if s > 0 else 0.0
                corp.capital += alloc

        self.portfolio = w.copy()
        return {"invested_amount": invested_amount, "weights": w.copy()}

    def to_dict(self):
        return {"id": self.id, "cash": self.cash, "portfolio": self.portfolio.tolist()}
