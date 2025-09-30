# gym_corp_biodiv_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any
from Env.cell import *
from Agents.corporation import Corporation
from Agents.investor import Investor
from Agents.reward import compute_all_rewards
from Env.species_dynamic import *
# action constants for readability
NO_ACTION = 0
EXPLOIT = 1
RESTORE = 2
GREEN_WASH = 3
RESILIENCE = 4

INV_NO_ACTION = 0
INV_INVEST = 1

class CorporateBiodiversityEnv(gym.Env):
    """
    Multi-agent environment for corporations and investors.
    - Uses dict interface: reset() -> obs_dict
                          step(actions_dict) -> obs_dict, reward_dict, done_dict, info_dict
    - actions_dict key is agent_id, e.g. 'corp_0' / 'inv_0', value is action dict corresponding to action space.

    TODO: Need to consider whether to affect cell's biodiversity attributes through disturbance, or directly affect cell's biodiversity attributes??

    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 grid_size = 100,
                 n_species = 25,
                 carrying_capacity = 25, 
                 disturbance = 0.0,
                 min_age=5, 
                 max_age=10, 
                 max_age_sort=True,
                 lat_steep=0.1,
                 disp_rate=0.45,
                 n_corporations: int = 3, 
                 n_investors: int = 2, 
                 max_steps: int = 1e6, 
                 half: bool = True,
                 birth_first: bool = True,
                 seed: int = 42):
        super().__init__()
        # environment state
        self.seed = int(seed)
        self.grid_size = int(grid_size)
        self.n_species = int(n_species)
        self.carrying_capacity = int(carrying_capacity)
        self.disturbance = float(disturbance)
        self.min_age = int(min_age)
        self.max_age = int(max_age)
        self.max_age_sort = bool(max_age_sort)
        self.disp_rate = float(disp_rate)
        self.lat_steep = float(lat_steep)
        self.half = bool(half)
        self.cell_id_n_coord = None
        self.d_matrix = None
        self.list_cells = None
        self.n_cells = grid_size * grid_size
        self.sdyn = None  # species dynamics placeholder
        self.birth_first = bool(birth_first)
        self.n_corporations = int(n_corporations)
        self.n_investors = int(n_investors)
        self.max_steps = int(max_steps)

        # create agents
        self.corporations = [Corporation(i, capital=100.0, biodiversity_score=1.0, resilience=0.0) for i in range(self.n_corporations)]
        self.investors = [Investor(i, cash=100.0, n_corporations=self.n_corporations) for i in range(self.n_investors)]

        # map agent ids
        self.corp_ids = [f"corp_{i}" for i in range(self.n_corporations)]
        self.inv_ids = [f"inv_{i}" for i in range(self.n_investors)]
        self.agent_ids = self.corp_ids + self.inv_ids

        # action & obs spaces dictionaries (for external use / agents)
        self.action_spaces: Dict[str, spaces.Space] = {}
        self.observation_spaces: Dict[str, spaces.Space] = {}

        # corp action: (action_type in [0..4], cell_index in [0..n_cells-1])
        # action_type: 0 - no action, 1 - exploit, 2 - restore, 3 - green wash, 4 - invest in resilience
        # cell_index: 0 to n_cells-1, which cell to act on, only one cell at a time
        ### TODO: Consider how many cells each corporation can select for exploit/restore, currently only allows selecting one cell
        ### Consider modifying action_space and step function to support multiple cells
        corp_action_space = spaces.Dict({
            "action_type": spaces.Discrete(5),
            "cell": spaces.Discrete(self.n_cells)
        })
        # investor action: (action_type: 0/1, weights: float vector length Nc)
        inv_action_space = spaces.Dict({
            "action_type": spaces.Discrete(2),
            "weights": spaces.Box(low=0.0, high=1.0, shape=(self.n_corporations,), dtype=np.float32)
        })

        # observation shapes:
        ### TODO: Define corporate observation space: can they observe other corporations' conditions,
        ### whether to observe all cell conditions or each corporation can only observe biodiversity and disturbance within a certain grid range
        ### Currently can only observe average disturbance and average n_individuals of all grids
        # corp obs: [capital, biodiv_score, resilience, mean_disturbance, mean_n_individuals]
        corp_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ### TODO: Define investor observation space
        # investor obs: [cash] + portfolio(Nc) + corp_capitals(Nc) + corp_biodiv(Nc)
        inv_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1 + self.n_corporations * 2 + self.n_corporations,), dtype=np.float32)
        # note: the shape above is 1 + Nc + Nc + Nc = 1 + 3*Nc; adjust if you change content.

        for aid in self.corp_ids:
            self.action_spaces[aid] = corp_action_space
            self.observation_spaces[aid] = corp_obs_space
        for aid in self.inv_ids:
            self.action_spaces[aid] = inv_action_space
            self.observation_spaces[aid] = inv_obs_space

        # parameters for ecological/economic rules (adjustable)
        self.params = {
            "exploit_gain_rate": 1.0,
            "exploit_disturb_increase": 15,
            "exploit_biodiv_loss": 0.02,
            "restore_cost": 5.0,
            "restore_effect": 5,
            "restore_biodiv_gain": 0.01,
            "greenwash_cost": 2.0,
            "greenwash_benefit": 0.1,
            "resilience_cost": 3.0,
            "resilience_gain": 0.05
        }

        # RNG
        self.np_random = np.random.default_rng(seed)
        self.step_count = 0

    def _get_mean_biodiversity(self):
        if self.n_cells == 0:
            return 0.0
        return float(np.mean([getattr(c, "shannon_div_idx", 0.0) for c in self.list_cells]))

    def _get_mean_disturbance(self):
        if self.n_cells == 0:
            return 0.0
        return float(np.mean([getattr(c, "disturbance", 0.0) for c in self.list_cells]))

    def reset(self, *, seed: int = None, options: dict = None):
        # re-seed generator if seed provided
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.step_count = 0

        # reset cells' states
        self.cell_id_n_coord = get_coordinates(self.grid_size)
        self.d_matrix = get_all_to_all_dist(self.grid_size)
        list_cells = init_cell_objects(self.cell_id_n_coord,
                                            self.d_matrix,
                                            self.n_species,
                                            self.carrying_capacity,
                                            self.disturbance,
                                            min_age=self.min_age,
                                            max_age=self.max_age,
                                            lat_steep=self.lat_steep,
                                            max_age_sort=self.max_age_sort
                                            )

        self.list_cells = init_species_population(n_species=self.n_species,
                                                list_cells=list_cells,
                                                cell_id_n_coord=self.cell_id_n_coord,
                                                cell_carrying_capacity=self.carrying_capacity,
                                                disp_rate=self.disp_rate,
                                                seed=self.seed,
                                                half=self.half,
                                            )
                # reset agents to initial values (could be randomized)
        self.sdyn = SpeciesDynamic(self.list_cells)
        self.sdyn.init_sp_sensitivities(seed = self.seed)
        self.sdyn.init_disturbance_generators()
        for corp in self.corporations:
            corp.reset()
        for inv in self.investors:
            inv.reset()
        # optionally reset cells' disturbances (Assume here list_cell already in initial state）
        # If you want to reset cell  some fields，can operate here
        return self._get_obs_dict()
    

    ### TODO: define what corporations observe
    # the following function is used to get the observation for a specific corporation
    def _get_obs_for_corp(self, corp_idx: int):
        corp = self.corporations[corp_idx]
        # the mean disturbance of the whole grid, reflect the overall anthropogenic disturbance
        mean_dist = self._get_mean_disturbance()
        # the mean number of individuals in the whole grid, reflect the overall biodiversity
        mean_bio_div = self._get_mean_biodiversity()
        obs = np.array([corp.capital, corp.biodiversity_score, corp.resilience, mean_dist, mean_bio_div], dtype=np.float32)
        return obs

    ### TODO: define what investors observe
    def _get_obs_for_inv(self, inv_idx: int):
        inv = self.investors[inv_idx]
        corp_capitals = np.array([c.capital for c in self.corporations], dtype=np.float32)
        corp_biodiv = np.array([c.biodiversity_score for c in self.corporations], dtype=np.float32)
        # obs layout: [cash, portfolio(Nc), corp_capitals(Nc), corp_biodiv(Nc)]
        obs = np.concatenate([[inv.cash], inv.portfolio.astype(np.float32), corp_capitals, corp_biodiv]).astype(np.float32)
        return obs

    ### TODO: integrate and summarizecorpandinv观察toobs，form global observation space
    def _get_obs_dict(self):
        obs = {}
        # "i" is the index, "aid" is the agent id string
        for i, aid in enumerate(self.corp_ids):
            obs[aid] = self._get_obs_for_corp(i)
        for i, aid in enumerate(self.inv_ids):
            obs[aid] = self._get_obs_for_inv(i)
        return obs

    def step(self, actions: Dict[str, Dict[str, Any]]):
        """
        actions: dict keyed by agent_id, each value matches agent's action_space.
        - each action is decided by the agent's policy (for demo, here is sampled randomly)
        - actions are applied in the order of agent IDs
        return: obs_dict, reward_dict, done_dict, info_dict
        """
        self.step_count += 1
        #------------------- Get the previous states (for rewards) ------------------- 
        ### TODO: acquire previous state
        # getting the mean disturbance before making actions
        prev_mean_dist = get_mean_disturbance(self.list_cells)
        prev_states = {
            "corporations": [(corp.capital, prev_mean_dist) for corp in self.corporations],
            "investors": []
        }
        ### TODO: how to calculate the Portfolio Value needs to be decided here and in reward
        #the previous portfolio value of each investor
        for inv in self.investors:
            # Calculateportfolio value
            portfolio_value = sum(inv.portfolio[i] * self.corporations[i].capital for i in range(self.n_corporations))
            prev_states["investors"].append(portfolio_value)
        #------------------- Make the actions ------------------- 
        # --- 1) execute corporations action ---
        for i, aid in enumerate(self.corp_ids):
            act = actions.get(aid, None)
            corp = self.corporations[i]
            if act is None:
                continue
            action_type = int(act.get("action_type", NO_ACTION))
            #choose target cell index, default to 0 if not provided, only one cell each exploitation
            # currently only supportsselectonecellperform operations，后续can以考虑support多个cell
            target_cell_idx = int(act.get("cell", 0)) if "cell" in act else 0
            # ensure target_cell_idx is within valid range 【0, n_cells-1】
            target_cell_idx = max(0, min(self.n_cells - 1, target_cell_idx)) if self.n_cells > 0 else None

            if action_type == NO_ACTION:
                pass
            elif action_type == EXPLOIT and target_cell_idx is not None:
                cell = self.list_cells[target_cell_idx]
                corp.apply_exploit(cell,
                                   gain_rate=self.params["exploit_gain_rate"],
                                   disturbance_increase=self.params["exploit_disturb_increase"],
                                   biodiv_loss=self.params["exploit_biodiv_loss"])
            elif action_type == RESTORE and target_cell_idx is not None:
                cell = self.list_cells[target_cell_idx]
                corp.apply_restore(cell,
                                   cost=self.params["restore_cost"],
                                   restore_amount=self.params["restore_effect"],
                                   biodiv_gain=self.params["restore_biodiv_gain"])
            elif action_type == GREEN_WASH:
                corp.green_wash(cost=self.params["greenwash_cost"], perceived_increase=self.params["greenwash_benefit"])
            elif action_type == RESILIENCE:
                corp.invest_resilience(cost=self.params["resilience_cost"], resilience_gain=self.params["resilience_gain"])
            else:
                # unknown action -> ignore
                pass

        # --- 2) execute investors action ---
        for i, aid in enumerate(self.inv_ids):
            act = actions.get(aid, None)
            inv = self.investors[i]
            if act is None:
                continue
            action_type = int(act.get("action_type", INV_NO_ACTION))
            if action_type == INV_NO_ACTION:
                pass
            elif action_type == INV_INVEST:
                # get weights, default to zero vector if not provided
                weights = act.get("weights", np.zeros(self.n_corporations, dtype=float))
                # ensure shape
                weights = np.array(weights, dtype=float).reshape(self.n_corporations)
                inv.invest(weights, corporations=self.corporations, normalize=True)
            else:
                pass

        # ------------------- Update Species distribution -------------------
        self.sdyn.update_disturbance_matrix(self.list_cells)
        self.sdyn.perform_species_dynamics(birth_first=self.birth_first,
                                           disp_rate=self.disp_rate)
        # ------------------- Calculateobservation、placeholder reward、done、info -------------------
        obs = self._get_obs_dict()
        ### TODO： think about how to definereward function
        reward_result = compute_all_rewards(self.corporations, self.investors, prev_states, self.list_cells)
        rewards = {}
        for i, aid in enumerate(self.corp_ids):
            rewards[aid] = reward_result["corporations"][i]
        for i, aid in enumerate(self.inv_ids):
            rewards[aid] = reward_result["investors"][i]
        ### TODO: think about eachagent什么时候toend state
        # done_flag: check if the agent has reached the end state
        done_flag = self.step_count >= self.max_steps
        dones = {aid: done_flag for aid in self.agent_ids}
        ### TODO: think about how to define eachagentinfo
        # infos can be empty dicts for now
        infos = {aid: {} for aid in self.agent_ids}

       


        return obs, rewards, dones, infos

    def render(self, mode="human"):
        # simple textThisrender and display several summary information
        print(f"Step {self.step_count}/{self.max_steps}")
        for i, corp in enumerate(self.corporations):
            print(f"Corp {i}: capital={corp.capital:.2f}, biodiv={corp.biodiversity_score:.3f}, resilience={corp.resilience:.3f}")
        for i, inv in enumerate(self.investors):
            print(f"Investor {i}: cash={inv.cash:.2f}, portfolio={inv.portfolio}")
        # show map disturbance summary
        disturbances = [getattr(c, "disturbance", 0.0) for c in self.list_cells]
        print(f"Map mean disturbance: {np.mean(disturbances):.3f}, max: {np.max(disturbances):.3f}")

    def close(self):
        pass


# minimal usage example (as scriptThisruntime)
# if __name__ == "__main__":
#     # 需要先定义one非常Simple CellClass used for demo
#     grid_size = 100
#     n_species = 25
#     carrying_capacity = 25

#     cell_id_n_coord = get_coordinates(grid_size)
#     d_matrix = get_all_to_all_dist(grid_size)
#     list_cells = init_cell_objects(cell_id_n_coord, d_matrix, n_species, carrying_capacity, lat_steep=0.1)
#     list_cells = init_species_population(
#         n_species=n_species,
#         list_cells=list_cells,
#         cell_id_n_coord=cell_id_n_coord,
#         cell_carrying_capacity=carrying_capacity,
#         disp_rate=0.45,
#         seed=42,
#     )


#     env = CorporateBiodiversityEnv(list_cell=list_cells, n_corporations=3, n_investors=2, max_steps=10)
#     obs = env.reset()
#     for t in range(5):
#         # random采样 actions
#         actions = {}
#         for aid in env.agent_ids:
#             actions[aid] = env.action_spaces[aid].sample()
#             # gym's Dict sample returns e.g. numpy types; convert to python types if needed
#             if isinstance(actions[aid], dict) and "weights" in actions[aid]:
#                 actions[aid]["weights"] = np.array(actions[aid]["weights"], dtype=float)
#         obs, rewards, dones, infos = env.step(actions)
#         env.render()
#         if all(dones.values()):
#             break
