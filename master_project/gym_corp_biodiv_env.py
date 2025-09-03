"""
Minimal Gym environment: Corporate Biodiversity (MVP)

File: gym_corp_biodiv_env.py
- A compact, single-file Gym environment implementing the simplified transitions
  and rewards discussed in the proposal conversation.
- Purpose: run a minimal, runnable prototype (10x10 grid, 3 corporations, 2 investors)
  that demonstrates state updates (H, D, T), corp actions (exploit/restore/greenwash/resilience),
  investor allocations, and reward calculation.

Usage:
    pip install gym numpy
    python gym_corp_biodiv_env.py

The script will run a short random-action demonstration when executed as main.

Note: this is a prototype for development and experimentation. It is intentionally
simple and easy to extend (e.g., continuous actions, PettingZoo multi-agent wrappers,
stable-baselines3 training scripts, richer ecological dynamics, audits, etc.).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pprint

# ---------------------- Helper functions ----------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def laplacian_flat(H, grid_size):
    # H: flat array of length grid_size**2
    H2 = H.reshape((grid_size, grid_size))
    # periodic boundary via np.roll for simplicity
    up = np.roll(H2, -1, axis=0)
    down = np.roll(H2, 1, axis=0)
    left = np.roll(H2, -1, axis=1)
    right = np.roll(H2, 1, axis=1)
    lap = up + down + left + right - 4.0 * H2
    return lap.reshape(-1)


# ---------------------- Environment ----------------------
class CorporateBiodiversityEnv(gym.Env):
    """Minimal Gym Env for Corporate impacts on biodiversity (MVP)

    Action space (Dict):
        - 'corp_actions': MultiDiscrete([5]*Nc)  # 0:noop,1:exploit,2:restore,3:greenwash,4:resilience
        - 'corp_targets': MultiDiscrete([grid_cells]*Nc)  # target cell index for exploit/restore
        - 'investor_actions': MultiDiscrete([Nc+1]*Ni)  # for each investor: 0..Nc-1 invest in corp, Nc means hold cash

    Observation (Dict): flattened arrays for simplicity:
        - 'H': Box(0,K0, shape=(grid_cells,))  # species abundance
        - 'D': Box(0,1, shape=(grid_cells,))   # anthropogenic disturbance
        - 'T': Box(0,10, shape=(grid_cells,))  # climate pressure
        - 'corp_capitals': Box(0,1e9, shape=(Nc,))
        - 'S_true': Box(0,1, shape=(Nc,))
        - 'S_perc': Box(0,1, shape=(Nc,))
        - 'investor_cash': Box(0,1e9, shape=(Ni,))
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 grid_size=10,
                 Nc=3,
                 Ni=2,
                 K0=100.0,
                 max_steps=200,
                 seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.grid_cells = grid_size * grid_size
        self.Nc = Nc
        self.Ni = Ni
        self.K0 = float(K0)
        self.max_steps = max_steps
        self.step_count = 0

        # --- Simple parameters (tunable) ---
        self.r = 0.3  # intrinsic growth
        self.kappa_D = 0.8
        self.kappa_T = 0.5
        self.m_D = 0.05
        self.m_T = 0.03
        self.gamma = 0.1  # diffusion strength
        self.phi = 0.05  # D natural decay
        self.beta_E = 0.12
        self.beta_R = 0.2
        self.eta_R = 1.0  # restore direct ecological gain

        # Economic parameters
        self.alpha = np.random.uniform(0.8, 1.2, size=(Nc,))  # production eff
        self.lambda_R = 0.1  # resilience benefit to revenue
        self.c_E = 1.0
        self.c_R = 2.0
        self.c_G = 0.5
        self.c_Res = 1.5
        self.c_fix = 0.1
        self.invest_amount = 50.0  # fixed per-investor per-step
        self.underline_C = 1.0  # bankruptcy threshold
        self.M = 1000.0  # bankruptcy penalty in reward

        # scoring / disclosure
        self.psi_D = 0.5
        self.psi_R = 0.2
        self.theta_q = 0.3  # disclosure noise/quality: lower => more truthful
        self.gamma_G = 2.0
        self.pi_q = 5.0  # greenwash penalty magnitude

        # reward weights
        self.chi = 0.05
        self.xi_R = 0.5
        self.upsilon = 0.2  # financing benefit

        # investor params
        self.Gamma_j = 0.02
        self.rho_j = 2.0
        self.eta_j = 0.01

        # initialize random
        self.np_random = np.random.RandomState(seed)

        # footprint w_i(c): for simplicity assign contiguous blocks per corp
        self.w = np.zeros((Nc, self.grid_cells), dtype=float)
        # create simple partition of grid cells among corporations
        cells_per = self.grid_cells // Nc
        for i in range(Nc):
            start = i * cells_per
            end = (i + 1) * cells_per if i < Nc - 1 else self.grid_cells
            self.w[i, start:end] = 1.0
        # normalize per-cell responsibility if overlapping: here non-overlap so fine

        # --- Action and observation spaces ---
        self.action_space = spaces.Dict({
            'corp_actions': spaces.MultiDiscrete([5] * Nc),
            'corp_targets': spaces.MultiDiscrete([self.grid_cells] * Nc),
            'investor_actions': spaces.MultiDiscrete([Nc + 1] * Ni),
        })

        obs_spaces = {
            'H': spaces.Box(low=0.0, high=self.K0 * 10.0, shape=(self.grid_cells,), dtype=np.float32),
            'D': spaces.Box(low=0.0, high=1.0, shape=(self.grid_cells,), dtype=np.float32),
            'T': spaces.Box(low=0.0, high=10.0, shape=(self.grid_cells,), dtype=np.float32),
            'corp_capitals': spaces.Box(low=0.0, high=1e9, shape=(Nc,), dtype=np.float32),
            'S_true': spaces.Box(low=0.0, high=1.0, shape=(Nc,), dtype=np.float32),
            'S_perc': spaces.Box(low=0.0, high=1.0, shape=(Nc,), dtype=np.float32),
            'investor_cash': spaces.Box(low=0.0, high=1e9, shape=(Ni,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # state variables (initialized in reset)
        self.H = None
        self.D = None
        self.T = None
        self.corp_capitals = None
        self.corp_R = None
        self.S_true = None
        self.S_perc = None
        self.investor_cash = None

        # bookkeeping
        self.last_investor_actions = None
        self.seed(seed)
        self.reset()

    # ---------- env API ----------
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def reset(self):
        self.step_count = 0
        # initialize fields
        self.H = np.clip(self.K0 * 0.5 + self.np_random.randn(self.grid_cells) * (0.05 * self.K0), 0.0, None)
        self.D = np.clip(self.np_random.rand(self.grid_cells) * 0.05, 0.0, 1.0)
        self.T = np.clip(self.np_random.rand(self.grid_cells) * 0.1, 0.0, 10.0)

        self.corp_capitals = np.full((self.Nc,), 500.0)
        self.corp_R = np.full((self.Nc,), 0.1)  # small initial resilience
        self.S_true = np.zeros((self.Nc,))
        self.S_perc = np.zeros((self.Nc,))
        self.investor_cash = np.full((self.Ni,), 200.0)

        self.last_investor_actions = np.full((self.Ni,), self.Nc, dtype=int)  # hold by default

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return {
            'H': self.H.astype(np.float32),
            'D': self.D.astype(np.float32),
            'T': self.T.astype(np.float32),
            'corp_capitals': self.corp_capitals.astype(np.float32),
            'S_true': np.clip(self.S_true, 0.0, 1.0).astype(np.float32),
            'S_perc': np.clip(self.S_perc, 0.0, 1.0).astype(np.float32),
            'investor_cash': self.investor_cash.astype(np.float32),
        }

    def step(self, action):
        self.step_count += 1
        info = {}

        # ---- parse actions ----
        corp_actions = np.array(action['corp_actions'], dtype=int)
        corp_targets = np.array(action['corp_targets'], dtype=int)
        investor_actions = np.array(action['investor_actions'], dtype=int)

        # build masks
        exploit_counts = np.zeros(self.grid_cells, dtype=float)
        restore_counts = np.zeros(self.grid_cells, dtype=float)
        corp_n_exploit = np.zeros(self.Nc, dtype=int)
        corp_n_restore = np.zeros(self.Nc, dtype=int)
        corp_greenwash_amount = np.zeros(self.Nc, dtype=float)
        corp_res_invest = np.zeros(self.Nc, dtype=float)

        for i in range(self.Nc):
            act = corp_actions[i]
            tgt = corp_targets[i] % self.grid_cells
            if act == 1:  # exploit
                exploit_counts[tgt] += 1.0
                corp_n_exploit[i] += 1
            elif act == 2:  # restore
                restore_counts[tgt] += 1.0
                corp_n_restore[i] += 1
            elif act == 3:  # greenwash
                corp_greenwash_amount[i] += 1.0
            elif act == 4:  # resilience invest
                corp_res_invest[i] += 1.0

        # ---- update D (disturbance) ----
        self.D = np.clip((1.0 - self.phi) * self.D + self.beta_E * exploit_counts - self.beta_R * restore_counts, 0.0, 1.0)

        # ---- update climate T (simple AR(1) noise) ----
        self.T = np.clip(0.95 * self.T + self.np_random.randn(self.grid_cells) * 0.01, 0.0, 10.0)

        # ---- compute K_t per cell and update H ----
        Kt = self.K0 * (1.0 - self.kappa_D * self.D) * (1.0 - self.kappa_T * self.T)
        Kt = np.clip(Kt, 1e-3, None)

        lap = laplacian_flat(self.H, self.grid_size)
        restore_effect = self.eta_R * restore_counts  # direct ecological gain per restore action

        H_next = self.H + self.r * self.H * (1.0 - (self.H / Kt)) - self.m_D * self.D * self.H - self.m_T * self.T * self.H + self.gamma * lap + restore_effect
        H_next = np.clip(H_next, 0.0, None)

        # count local extinctions
        extinct_mask = (self.H > 1e-6) & (H_next <= 1e-6)
        n_extinctions = int(np.sum(extinct_mask))

        self.H = H_next

        # ---- company revenues and costs ----
        Rev = np.zeros(self.Nc, dtype=float)
        Cost = np.zeros(self.Nc, dtype=float)
        I_inflows = np.zeros(self.Nc, dtype=float)

        for i in range(self.Nc):
            # revenue: sum across exploited targets for corp i
            # each exploit action was counted once per corp and targeted cell in corp_targets
            if corp_n_exploit[i] > 0:
                # find target index for this corp (we used corp_targets)
                # as simplified model, revenue equals sum over corp_n_exploit occurrences
                # find all indices j where corp_actions[j]==1 and target==cell
                # simpler: use corp_targets[i] (only one target per corp per step in MVP)
                tgt = corp_targets[i] % self.grid_cells
                Rev[i] = corp_n_exploit[i] * (self.alpha[i] * self.H[tgt] + self.lambda_R * self.corp_R[i])
            # costs
            Cost[i] = self.c_E * corp_n_exploit[i] + self.c_R * corp_n_restore[i] + self.c_G * corp_greenwash_amount[i] + self.c_Res * corp_res_invest[i] + self.c_fix

        # ---- investor allocations: simple fixed-amount invest if choosing a corp ----
        for j in range(self.Ni):
            choice = investor_actions[j]
            if choice < self.Nc:
                amount = min(self.invest_amount, self.investor_cash[j])
                I_inflows[choice] += amount
                self.investor_cash[j] -= amount
            else:
                # hold cash
                pass

        # ---- update capitals ----
        F = np.zeros(self.Nc, dtype=float)  # regulatory fines (none in baseline)
        C_next = self.corp_capitals + Rev - Cost + I_inflows - F

        # handle bankruptcies (flag and set capital to zero)
        bankrupt = C_next < self.underline_C
        for i in range(self.Nc):
            if bankrupt[i]:
                C_next[i] = 0.0

        self.corp_capitals = C_next

        # ---- update S_true and S_perc ----
        for i in range(self.Nc):
            # true score: 1 - psi_D * sum w_i(c) * D(c) + psi_R * restores
            true_score = 1.0 - self.psi_D * float(np.dot(self.w[i], self.D)) + self.psi_R * float(corp_n_restore[i])
            self.S_true[i] = np.clip(true_score, 0.0, 1.0)
            # perceived score: mix with greenwash signal
            gw_signal = sigmoid(self.gamma_G * corp_greenwash_amount[i])
            perc = (1.0 - self.theta_q) * self.S_true[i] + self.theta_q * gw_signal + self.np_random.randn() * 0.01
            self.S_perc[i] = np.clip(perc, 0.0, 1.0)

        # ---- update resilience R_i ----
        self.corp_R = np.clip((1.0 - 0.05) * self.corp_R + 0.3 * corp_res_invest, 0.0, 10.0)

        # ---- compute rewards (per-corp and per-investor) ----
        corp_rewards = np.zeros(self.Nc, dtype=float)
        for i in range(self.Nc):
            ecology_penalty = self.chi * float(np.dot(self.w[i], self.D * self.H)) * (1.0 - self.xi_R * self.corp_R[i])
            greenwash_pen = self.pi_q * max(0.0, self.S_perc[i] - self.S_true[i])
            corp_rewards[i] = (Rev[i] - Cost[i]) - ecology_penalty - greenwash_pen + self.upsilon * I_inflows[i]
            if bankrupt[i]:
                corp_rewards[i] -= self.M

        inv_rewards = np.zeros(self.Ni, dtype=float)
        for j in range(self.Ni):
            choice = investor_actions[j]
            if choice < self.Nc:
                # realized return approx (Rev - Cost)/capital before investment
                denom = max(1.0, self.corp_capitals[choice])
                ret = (Rev[choice] - Cost[choice]) / denom
                inv_rewards[j] = ret - 0.5 * self.Gamma_j * (ret ** 2) - self.rho_j * max(0.0, (self.S_perc[choice] - self.S_true[choice])) - self.eta_j * (1.0 if self.last_investor_actions[j] != choice else 0.0)
            else:
                inv_rewards[j] = 0.0

        # bookkeeping last investor action
        self.last_investor_actions = investor_actions.copy()

        # aggregate reward (Gym requires a single scalar reward) -> we return sum of corp rewards
        reward = float(np.sum(corp_rewards))

        # prepare info with per-agent rewards and diagnostics
        info['rewards'] = {'corp': corp_rewards.tolist(), 'investor': inv_rewards.tolist()}
        info['n_extinctions'] = n_extinctions
        info['bankrupt'] = bankrupt.tolist()

        # done condition
        done = bool(self.step_count >= self.max_steps)

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        # simple text rendering
        print(f"Step {self.step_count} | Total H: {self.H.sum():.1f} | Avg D: {self.D.mean():.3f} | Corp caps: {np.round(self.corp_capitals,2)} | S_perc: {np.round(self.S_perc,3)}")


# ---------------------- Demo run ----------------------
if __name__ == '__main__':
    env = CorporateBiodiversityEnv(grid_size=6, Nc=3, Ni=2, seed=42, max_steps=50)
    obs = env.reset()
    print("Reset obs keys:")
    pprint.pprint({k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()})

    for t in range(20):
        action = {
            'corp_actions': env.action_space.spaces['corp_actions'].sample(),
            'corp_targets': env.action_space.spaces['corp_targets'].sample(),
            'investor_actions': env.action_space.spaces['investor_actions'].sample(),
        }
        obs, reward, done, info = env.step(action)
        env.render()
        if t % 5 == 0:
            pprint.pprint(info)
        if done:
            break

    print('Demo finished')
