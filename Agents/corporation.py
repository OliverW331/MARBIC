# Agents/corporation.py
import numpy as np
from typing import Any

class Corporation:
    """
    简单的 Corporation 类，封装资本、biodiversity score、resilience，
    并提供几个 action helper（不直接修改环境全局状态以外的东西除非传入 cell）。
    """

    def __init__(self, corp_id: int, capital: float = 100.0, biodiversity_score: float = 1.0, resilience: float = 0.0):
        self.id = int(corp_id)
        self.capital = float(capital)
        self.biodiversity_score = float(biodiversity_score)
        self.resilience = float(resilience)

    def reset(self):
        self.capital = 100.0
        self.biodiversity_score = 1.0
        self.resilience = 0.0

    def apply_exploit(self, cell: Any, gain_rate: float = 1.0, disturbance_increase: float = 0.1, biodiv_loss: float = 0.05):
        """
        对一个 cell 进行剥削：增加自身资本、增加 cell.disturbance、降低自身生物多样性评分（受 resilience 缓冲）
        cell 预期具有属性：n_individuals, disturbance, n_species
        """
        ### TODO: 修改资本增加的方式
        gain = gain_rate * float(getattr(cell, "n_individuals", 0.0))
        self.capital += gain

        ### TODO: 修改增加disturbance变化的方式
        # 增加 disturbance，clamp 到 [0,1]
        # disturbance is changed on cell level
        # new_dist = min(1.0, getattr(cell, "disturbance", 0.0) + disturbance_increase)
        new_dist = getattr(cell, "disturbance", 0.0) + disturbance_increase
        cell.disturbance = new_dist

        ### TODO: 因为resilience应该减小的是公司disturbance对biodiversity score的影响，而不是直接减小biodiversity score
        # biodiversity score 下降，被 resilience 部分抵消
        loss = biodiv_loss * (1.0 - self.resilience)
        self.biodiversity_score = max(0.0, self.biodiversity_score - loss)

        ### TODO: 思考如何影响选中cell的biodiversity和species histogram

        return {"gain": gain, "new_disturbance": new_dist, "biodiv_loss": loss}

    def apply_restore(self, cell: Any, cost: float = 5.0, restore_amount: float = 0.2, biodiv_gain: float = 0.02):
        """
        对一个 cell 恢复：付出资本，降低 cell.disturbance，提高公司 biodiv score
        """
        ### TODO: 修改减少资本的方式
        self.capital = max(0.0, self.capital - cost)
        
        ### TODO: 修改减少disturbance的方式
        # cell.disturbance = max(0.0, getattr(cell, "disturbance", 0.0) - restore_amount)
        cell.disturbance = getattr(cell, "disturbance", 0.0) - restore_amount
        ### TODO: 思考如何影响选中cell的biodiversity和species histogram
        self.biodiversity_score += biodiv_gain
        return {"cost": cost, "new_disturbance": cell.disturbance, "biodiv_gain": biodiv_gain}

    def green_wash(self, cost: float = 2.0, perceived_increase: float = 0.1):
        """
        公关（green wash）：不改变真实环境，只改变公司感知 biodiv score（用于模拟欺骗或市场表演）
        """
        ### TODO: 修改如何减少资本的方式
        self.capital = max(0.0, self.capital - cost)

        ### TODO: 思考是直接影响真实biodiversity score还是perceived biodiversity score
        ### TODO: 修改如何增加biodiversity score的方式
        self.biodiversity_score += perceived_increase
        return {"cost": cost, "perceived_increase": perceived_increase}

    def invest_resilience(self, cost: float = 3.0, resilience_gain: float = 0.05):
        """
        投资于公司的resilience: 付出资本，提高公司对于生物多样性减少的抵抗力
        """
        ### TODO: 修改如何减少资本的方式
        self.capital = max(0.0, self.capital - cost)
        
        ### TODO: 修改增加resilience的方式
        self.resilience += resilience_gain
        return {"cost": cost, "resilience_gain": resilience_gain}

    def to_dict(self):
        return {"id": self.id, "capital": self.capital, "biodiversity_score": self.biodiversity_score, "resilience": self.resilience}
