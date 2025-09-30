# Agents/corporation.py
import numpy as np
from typing import Any

class Corporation:
    """
    Simple Corporation class，encapsulates capital、biodiversity score、resilience，
    and provides several action helper（does not directly modify things other than environment global state unless passed in cell）。
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
        for one cell exploit：增加自身资This、增加 cell.disturbance、降低自身生物diversity评分（受 resilience 缓冲）
        cell 预期具有attributes：n_individuals, disturbance, n_species
        """
        ### TODO: 修改资This增加方式
        gain = gain_rate * float(getattr(cell, "n_individuals", 0.0))
        self.capital += gain

        ### TODO: 修改增加disturbancechange方式
        # 增加 disturbance，clamp to [0,1]
        # disturbance is changed on cell level
        # new_dist = min(1.0, getattr(cell, "disturbance", 0.0) + disturbance_increase)
        new_dist = getattr(cell, "disturbance", 0.0) + disturbance_increase
        cell.disturbance = new_dist

        ### TODO: Because resilience should reduce the impact of corporate disturbance on biodiversity score, not directly reduce biodiversity score
        # biodiversity score decreases, partially offset by resilience
        loss = biodiv_loss * (1.0 - self.resilience)
        self.biodiversity_score = max(0.0, self.biodiversity_score - loss)

        ### TODO: Consider how to affect the biodiversity and species histogram of selected cell

        return {"gain": gain, "new_disturbance": new_dist, "biodiv_loss": loss}

    def apply_restore(self, cell: Any, cost: float = 5.0, restore_amount: float = 0.2, biodiv_gain: float = 0.02):
        """
        Restore a cell: pay capital, reduce cell.disturbance, increase corporate biodiv score
        """
        ### TODO: Modify the way to reduce capital
        self.capital = max(0.0, self.capital - cost)
        
        ### TODO: Modify the way to reduce disturbance
        # cell.disturbance = max(0.0, getattr(cell, "disturbance", 0.0) - restore_amount)
        cell.disturbance = getattr(cell, "disturbance", 0.0) - restore_amount
        ### TODO: Consider how to affect the biodiversity and species histogram of selected cell
        self.biodiversity_score += biodiv_gain
        return {"cost": cost, "new_disturbance": cell.disturbance, "biodiv_gain": biodiv_gain}

    def green_wash(self, cost: float = 2.0, perceived_increase: float = 0.1):
        """
        Public relations (green wash): does not change real environment, only changes corporate perceived biodiv score (for simulating deception or market performance)
        """
        ### TODO: Modify how to reduce capital
        self.capital = max(0.0, self.capital - cost)

        ### TODO: Consider whether to directly affect real biodiversity score or perceived biodiversity score
        ### TODO: Modify how to increase biodiversity score
        self.biodiversity_score += perceived_increase
        return {"cost": cost, "perceived_increase": perceived_increase}

    def invest_resilience(self, cost: float = 3.0, resilience_gain: float = 0.05):
        """
        Invest in corporate resilience: pay capital, improve corporate resistance to biodiversity reduction
        """
        ### TODO: Modify how to reduce capital
        self.capital = max(0.0, self.capital - cost)
        
        ### TODO: Modify how to increase resilience
        self.resilience += resilience_gain
        return {"cost": cost, "resilience_gain": resilience_gain}

    def to_dict(self):
        return {"id": self.id, "capital": self.capital, "biodiversity_score": self.biodiversity_score, "resilience": self.resilience}
