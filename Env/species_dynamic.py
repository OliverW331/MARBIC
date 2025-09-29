'''
Simulate the dynamic of species population
- how the population grow
- how they expand on the map
- how the population decrease
    - three types of disturbance: D disturbance, B disturbance, K disturbance
        - D disturbance: increase the mortality rate
        - B disturbance: decrease the birth rate
        - K disturbance: decrease the carrying capacity for each cell
    - disturbance has the range of [0,1]
    - caused by random factor
出生扩散死亡代码实现的过程
    -循环每个cell（顺序随机），对每个cell里面的每个物种进行出生作为births，里面记录了每个出生的物种，总长度为出生量
    -将births打乱顺序，依次处理每个出生的物种，根据扩散概率选择一个cell进行扩散，如果该cell有空间就放进去，否则继续选择下一个cell，直到放进去或者尝试了所有cell（一般不会到达这个程度，这样的话说明整个环境都没有空间了）
    -循环完births应该就处理好了出生和扩散，然后再循环每个cell，对每个cell里面的每个物种进行死亡处理    
'''

from matplotlib.pylab import seed
from Disturbance.disturbance import Disturbance
import numpy as np
import matplotlib.pyplot as plt

### helper functions for species dynamics
def species_birth(list_cells, n_species, birth_rate_matrix):
    """
    Collect all new births as a list of (species_id, src_cell_idx).
    """
    grid_size = int(len(list_cells) ** 0.5)
    births = []
    for species_id in range(n_species):
        n_array = np.array([cell.species_hist[species_id] for cell in list_cells])
        birth_rates = birth_rate_matrix[:, :, species_id].reshape(-1)
        n_births = np.round(n_array * birth_rates).astype(int)
        idxs = np.arange(len(list_cells))
        births.extend([(species_id, idx) for idx, count in zip(idxs, n_births) for _ in range(count)])
    return births

def species_disperse(list_cells, births, disp_rate=1.0, seed=42):
    """
    Shuffle births, for each birth, try to disperse to a cell with space.
    """
    rng = np.random.default_rng(seed)
    n_cells = len(list_cells)
    aval_space = np.array([c.available_space for c in list_cells])
    print(f"[DEBUG] Initial available_space: {aval_space}")
    rng.shuffle(births)
    print(f"[DEBUG] births count: {len(births)}")
    for i, (species_id, src_idx) in enumerate(births): 
        cell = list_cells[src_idx]
        if aval_space[src_idx] > 0:
            cell.species_hist[species_id] += 1
            cell.sp_age_dict[species_id][0] += 1
            aval_space[src_idx]  = max(aval_space[src_idx] - 1, 0)
            # print(f"[DEBUG] Birth {i}: species {species_id} stays in cell {src_idx}, available_space now {aval_space[src_idx]}")
            continue
        aval_space[src_idx] = 0
        if np.sum(aval_space) == 0:
            continue
        disp = 1.0 / disp_rate
        disp_probability = disp * np.exp(-disp * cell.dist_matrix)
        sampling_prob = disp_probability * aval_space
        candidate_cells = np.where(aval_space > 0)[0]
        # check the structure of candidate_cells and their available_space
        if len(candidate_cells) == 0:
            continue
        probs = sampling_prob[candidate_cells]
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            continue
        # randomly choose one cell from candidate_cells with probability proportional to probs
        selected_cell = rng.choice(candidate_cells, p=probs / probs_sum)
        # or choose the one with the highest probability
        # selected_cell = candidate_cells[np.argmax(probs)]
        list_cells[selected_cell].species_hist[species_id] += 1
        list_cells[selected_cell].sp_age_dict[species_id][0] += 1
        aval_space[selected_cell] = max(aval_space[selected_cell] - 1, 0)
        # print(f"[DEBUG] Birth {i}: species {species_id} dispersed to cell {selected_cell}, available_space now {aval_space[selected_cell]}")
    # 结束后打印每个cell的总个体数
    # for idx, cell in enumerate(list_cells):
    #     print(f"[DEBUG] Cell {idx}: n_individuals={cell.n_individuals}, carrying_capacity={cell.carrying_capacity}")

def species_death(list_cells, n_species, death_rate_matrix):
    grid_size = int(len(list_cells) ** 0.5)
    for species_id in range(n_species):
        death_rates = death_rate_matrix[:, :, species_id].reshape(-1)
        # death_rates.shape = (n_cells,)
        for idx, cell in enumerate(list_cells):
            age_arr = cell.sp_age_dict[species_id]
            # 按年龄段批量死亡
            deaths_per_age = np.round(age_arr * death_rates[idx]).astype(int)
            age_arr[:] = np.maximum(0, age_arr - deaths_per_age)
            # 同步 species_hist
            cell.species_hist[species_id] = age_arr.sum()

### helper functions for visualization
def plot_carrying_capacity_vs_exploitation(disturbance_obj):
    f = disturbance_obj.f
    K = disturbance_obj.transform_to_K_disturbance()
    plt.figure(figsize=(6,4))
    plt.scatter(f.flatten(), K.flatten(), alpha=0.5)
    plt.xlabel("Effect of Exploitation (f)")
    plt.ylabel("Carrying Capacity Factor (K)")
    plt.title("Carrying Capacity vs Effect of Exploitation")
    plt.grid(True)
    plt.show()

def plot_birth_rate_vs_exploitation(disturbance_obj):
    f = disturbance_obj.f
    birth_rate = disturbance_obj.transform_to_B_disturbance()
    plt.figure(figsize=(6,4))
    plt.scatter(f.flatten(), birth_rate.flatten(), alpha=0.5)
    plt.xlabel("Effect of Exploitation (f)")
    plt.ylabel("Birth Rate Factor")
    plt.title("Birth Rate vs Effect of Exploitation")
    plt.grid(True)
    plt.show()

def plot_death_rate_vs_exploitation(disturbance_obj):
    f = disturbance_obj.f
    death_rate = disturbance_obj.transform_to_D_disturbance()
    plt.figure(figsize=(6,4))
    plt.scatter(f.flatten(), death_rate.flatten(), alpha=0.5)
    plt.xlabel("Effect of Exploitation (f)")
    plt.ylabel("Death Rate Factor")
    plt.title("Death Rate vs Effect of Exploitation")
    plt.grid(True)
    plt.show()

def plot_all_effects(disturbance_obj):
    f = disturbance_obj.f.flatten()
    K = disturbance_obj.transform_to_K_disturbance().flatten()
    birth_rate = disturbance_obj.transform_to_B_disturbance().flatten()
    death_rate = disturbance_obj.transform_to_D_disturbance().flatten()
    plt.figure(figsize=(7,5))
    plt.plot(f, K, 'o', label='Carrying Capacity', alpha=0.5)
    plt.plot(f, birth_rate, 'o', label='Birth Rate', alpha=0.5)
    plt.plot(f, death_rate, 'o', label='Death Rate', alpha=0.5)
    plt.xlabel("Effect of Exploitation (f)")
    plt.ylabel("Factor Value")
    plt.title("All Effects vs Effect of Exploitation")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_species_difference(list_cell_before, list_cell_after, species_id):
    grid_size = int(np.sqrt(len(list_cell_before)))
    hist_before = np.array([cell.species_hist[species_id] for cell in list_cell_before]).reshape((grid_size, grid_size))
    hist_after = np.array([cell.species_hist[species_id] for cell in list_cell_after]).reshape((grid_size, grid_size))
    diff = hist_after - hist_before

    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(label='Change in Population')
    plt.title(f'Species {species_id} Population Change')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_carrying_capacity_map(list_cells):
    grid_size = int(np.sqrt(len(list_cells)))
    carrying_capacities = np.array([cell.carrying_capacity for cell in list_cells]).reshape((grid_size, grid_size))
    plt.figure(figsize=(8, 6))
    plt.imshow(carrying_capacities, cmap='viridis', origin='lower')
    plt.colorbar(label='Carrying Capacity')
    plt.title('Carrying Capacity of Each Cell')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

class SpeciesDynamic:
    def __init__(self, list_cell):
        self.list_cell = list_cell
        self.grid_size = int(len(list_cell) ** 0.5)
        self.n_cells = len(list_cell)
        self.n_species = len(list_cell[0].species_hist)
        # if list_cell is just initialized, disturbance_matrix will be all zeros
        # each c.disturbance recorded how many times this cell has been exploited
        self.disturbance_matrix = np.array([c.disturbance for c in list_cell]).reshape((self.grid_size, self.grid_size))
        # the sensitivity of each species to D and B disturbance, shape = (n_species,)
        # larger value means more sensitive
        ### TODO: 没有包括K disturbance sensitivity，因为K disturbance是直接影响cell的carrying capacity
        self.D_disturbance_sensitivity = None
        self.B_disturbance_sensitivity = None
        self.K_disturbance_sensitivity = None
        # Disturbance generator object
        self.D_disturbance_generator= None
        self.B_disturbance_generator= None
        self.K_disturbance_generator= None
        # Effect matrices
        # D and B effect of exploitation on each cell and species, shape = (grid_size, grid_size, n_species)
        self.effect_of_D_exploitation = None
        self.effect_of_B_exploitation = None
        # self.effect_of_K_exploitation = None
        # These three disturbance matrices are all in the range of [0,1]
        self.K_dist_cell = None
        # Store baseline carrying capacity for each cell
        self.baseline_carrying_capacity = np.array([cell.carrying_capacity for cell in list_cell])
        # The D, B disturbance matrices for each cell and species, shape = (grid_size, grid_size, n_species)
        self.D_dist_cell_sp = None
        self.B_dist_cell_sp = None
    

    def init_sp_sensitivities(self, seed=42):
        np.random.seed(seed)
        self.D_disturbance_sensitivity = np.zeros(self.n_species) + np.random.random(self.n_species)
        self.B_disturbance_sensitivity = np.zeros(self.n_species) + np.random.random(self.n_species)

    def init_disturbance_generators(self, mode_D="exponential", mode_B="exponential", mode_K="exponential",
                                    alpha=0.1, beta=0.05, gamma=0.009758, k=1.0, f0=10.0):
        self.effect_of_D_exploitation = np.einsum('ij,k->ijk', self.disturbance_matrix, self.D_disturbance_sensitivity)
        self.D_disturbance_generator = Disturbance(self.effect_of_D_exploitation, alpha=alpha, beta=beta, gamma=gamma,
                                       k=k, f0=f0, D_mode=mode_D, B_mode=mode_B, K_mode=mode_K)
        self.effect_of_B_exploitation = np.einsum('ij,k->ijk', self.disturbance_matrix, self.B_disturbance_sensitivity)
        self.B_disturbance_generator = Disturbance(self.effect_of_B_exploitation, alpha=alpha, beta=beta, gamma=gamma,
                                       k=k, f0=f0, D_mode=mode_D, B_mode=mode_B, K_mode=mode_K)
        self.K_disturbance_generator = Disturbance(self.disturbance_matrix, alpha=alpha, beta=beta, gamma=gamma,
                                       k=k, f0=f0, D_mode=mode_D, B_mode=mode_B, K_mode=mode_K)

    def update_disturbance_matrix(self, list_cell):
        # update the disturbance matrix from the provided new list of cells
        self.list_cell = list_cell
        self.disturbance_matrix = np.array([c.disturbance for c in list_cell]).reshape((self.grid_size, self.grid_size))

    def update_D_disturbance_generator(self):
        # update the effect_of_exploitation matrix in the disturbance generator with the current disturbance_matrix
        self.effect_of_D_exploitation = np.einsum('ij,k->ijk', self.disturbance_matrix, self.D_disturbance_sensitivity)
        self.D_disturbance_generator.f = self.effect_of_D_exploitation

    def update_B_disturbance_generator(self):
        self.effect_of_B_exploitation = np.einsum('ij,k->ijk', self.disturbance_matrix, self.B_disturbance_sensitivity)
        self.B_disturbance_generator.f = self.effect_of_B_exploitation

    def update_K_disturbance_generator(self):
        # update the K disturbance generator with current disturbance_matrix
        self.K_disturbance_generator.f = self.disturbance_matrix

    def update_K_dist_cell(self):
        self.K_dist_cell = self.K_disturbance_generator.transform_to_K_disturbance()

    def update_D_dist_cell_sp(self):
        self.D_dist_cell_sp = self.D_disturbance_generator.transform_to_D_disturbance()
        #soft max normalization along species axis
        # exp_D = np.exp(self.D_dist_cell_sp)
        # self.D_dist_cell_sp = exp_D / np.sum(exp_D, axis=2, keepdims=True)
        # clip to [0,1]
        self.D_dist_cell_sp = np.clip(self.D_dist_cell_sp, 0.0, 1.0)
    def update_B_dist_cell_sp(self):
        self.B_dist_cell_sp = self.B_disturbance_generator.transform_to_B_disturbance()
        self.B_dist_cell_sp = np.clip(self.B_dist_cell_sp, 0.0, 1.0)
    def update_cell_carrying_capacity(self):
        # 确保K_dist_cell已更新
        if self.K_dist_cell is None:
            self.update_K_dist_cell()
        
        for i, cell in enumerate(self.list_cell):
            x = i // self.grid_size
            y = i % self.grid_size
            # 使用基准carrying capacity而不是当前值，避免复合增长
            cell.carrying_capacity = int(self.K_dist_cell[x, y] * self.baseline_carrying_capacity[i])
            ### after updating carrying capacity, some places may have n_individuals > carrying_capacity, need to adjust
            # how to adjust? randomly kill individuals until n_individuals <= carrying_capacity
            if cell.n_individuals > cell.carrying_capacity:
                excess = cell.n_individuals - cell.carrying_capacity
                # randomly kill individuals according to species_hist, species with more individuals have higher chance to die
                while excess > 0:
                    species_id = np.random.choice(np.where(cell.species_hist > 0)[0], p=cell.species_hist[cell.species_hist > 0] / cell.species_hist.sum())
                    cell.species_hist[species_id] -= 1
                    cell.sp_age_dict[species_id][0] = max(0, cell.sp_age_dict[species_id][0] - 1)
                    excess -= 1


    def update_sp_age_dict(self):
        """
        对所有 cell 的所有物种进行年龄增长和最大年龄死亡。
        """
        for cell in self.list_cell:
            for species_id, age_arr in cell.sp_age_dict.items():
                # 更新 species_hist，减去最大年龄死亡的个体数
                cell.species_hist[species_id] -= age_arr[-1]
                # 年龄增长：所有个体年龄+1，最大年龄的个体死亡
                age_arr[1:] = age_arr[:-1]
                age_arr[0] = 0  # 新出生后再加
            cell.species_hist = np.array([arr.sum() for arr in cell.sp_age_dict.values()])

        

    def perform_species_dynamics(self, birth_first=True, disp_rate=0.45):
        """
        Simulate one step of species population dynamics for all species.
        - birth_first: if True, do birth -> disperse -> death; else do death -> birth -> disperse
        - birth_rate, death_rate, carrying_capacity are dynamically affected by disturbance and sensitivity
        """
        self.update_D_disturbance_generator()
        self.update_B_disturbance_generator()
        self.update_K_disturbance_generator()
        self.update_D_dist_cell_sp()
        self.update_B_dist_cell_sp()
        self.update_K_dist_cell()
        self.update_cell_carrying_capacity()
        self.update_sp_age_dict()
        ### after updating carrying capacity, some places may have n_individuals > carrying_capacity
        birth_rate_matrix = self.B_dist_cell_sp  # shape: (grid_size, grid_size, n_species)
        death_rate_matrix = self.D_dist_cell_sp  # shape: (grid_size, grid_size, n_species)

        if birth_first:
            births = species_birth(self.list_cell, self.n_species, birth_rate_matrix)
            species_disperse(self.list_cell, births, disp_rate)
            species_death(self.list_cell, self.n_species, death_rate_matrix)
        else:
            species_death(self.list_cell, self.n_species, death_rate_matrix)
            births = species_birth(self.list_cell, self.n_species, birth_rate_matrix)
            species_disperse(self.list_cell, births, disp_rate)
    

