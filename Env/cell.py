import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numba import njit
import copy

# -------------------------------------------------
# print and emoji
# -------------------------------------------------
def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def get_nature_emoji():
    nature = np.array(
        [
            u"\U0001F400",
            u"\U0001F439",
            u"\U0001F430",
            u"\U0001F407",
            u"\U0001F43F",
            u"\U0001F994",
            u"\U0001F40F",
            u"\U0001F411",
            u"\U0001F410",
            u"\U0001F42A",
            u"\U0001F42B",
            u"\U0001F999",
            u"\U0001F992",
            u"\U0001F418",
            u"\U0001F98F",
            u"\U0001F993",
            u"\U0001F98C",
            u"\U0001F42E",
            u"\U0001F402",
            u"\U0001F403",
            u"\U0001F404",
            u"\U0001F405",
            u"\U0001F406",
            u"\U0001F989",
            u"\U0001F99C",
            u"\U0001F40A",
            u"\U0001F422",
            u"\U0001F98E",
            u"\U0001F40D",
            u"\U0001F995",
            u"\U0001F996",
            u"\U0001F433",
            u"\U0001F40B",
            u"\U0001F42C",
            u"\U0001F41F",
            u"\U0001F420",
            u"\U0001F421",
            u"\U0001F988",
            u"\U0001F419",
            u"\U0001F41A",
            u"\U0001F40C",
            u"\U0001F98B",
            u"\U0001F41B",
            u"\U0001F41C",
            u"\U0001F41D",
            u"\U0001F41E",
            u"\U0001F997",
            u"\U0001F577",
            u"\U0001F982",
            u"\U0001F99F",
            u"\U0001F9A0",
            u"\U0001F331",
            u"\U0001F332",
            u"\U0001F333",
            u"\U0001F334",
            u"\U0001F335",
            u"\U0001F33E",
            u"\U0001F33F",
        ]
    )
    return np.random.choice(nature)



# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def get_all_to_all_dist(n_cells):
    x = np.repeat(np.arange(n_cells), n_cells)
    y = np.tile(np.arange(n_cells), n_cells)
    coords = np.column_stack((x, y))  # shape = (n_cells^2, 2)

    # Calculate pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix

def get_coordinates(n_cells):
    """
    Return all cell  (id, x, y) coordinate array
    return coords in the format of (id, x, y)
    """
    ids = np.arange(n_cells * n_cells)
    x = np.repeat(np.arange(n_cells), n_cells)
    y = np.tile(np.arange(n_cells), n_cells)
    coords = np.column_stack((ids, x, y))
    return coords

def generate_weibull_abundance(n_species, weibull_k=0.75, scale_factor=1, seed=42):
    """
    Generate total abundance for each species using Weibull distribution.
    """
    rng = np.random.default_rng(seed)
    rel_abundance = rng.weibull(weibull_k, n_species + int(0.10 * n_species)) * scale_factor
    rel_abundance = np.sort(rel_abundance)[::-1][:n_species]  # descending order
    rel_frequencies = rel_abundance / np.sum(rel_abundance)
    return rel_frequencies

def sample_species_individuals(n_individuals, rel_frequencies, seed=42):
    """
    Sample species individuals based on their relative frequencies.
    sampled_individuals:                a sampling of the individuals based on their relative frequencies
    n_individuals_per_species_full:     the full count of individuals per species
    """
    species_id_array = np.arange(len(rel_frequencies))
    rng = np.random.default_rng(seed)
    sampled_individuals = rng.choice(species_id_array, size=n_individuals, p=rel_frequencies, replace=True)
    n_individuals_per_species = np.unique(sampled_individuals, return_counts=True)
    n_individuals_per_species_full = np.zeros(len(rel_frequencies), dtype=int)
    n_individuals_per_species_full[n_individuals_per_species[0]] = n_individuals_per_species[1]
    return sampled_individuals, n_individuals_per_species_full

def get_mean_disturbance(list_cell):
    mean_disturbance = np.mean([getattr(cell, "disturbance", 0.0) for cell in list_cell])
    return mean_disturbance

def rnd_sp_max_age(n_species, min_age=2, max_age=10, sort=False, seed=42):
    np.random.seed(seed)
    sp_max_age = np.random.randint(min_age, max_age+1, size=n_species)
    if sort:
        sp_max_age.sort()
    return sp_max_age


# -------------------------------------------------
# CellClass
# -------------------------------------------------
class CellClass:
    def __init__(
        self,
        coord,
        cell_id,
        dist_matrix,
        species_hist,
        sp_age_dict,
        carrying_capacity,
        disturbance=0,
        protection=0,
        pathogen=0,
        lat_steepness=0.1,
    ):
        self.coord = coord  # array of two integers
        self.id = cell_id
        self.carrying_capacity = carrying_capacity
        self.species_hist = species_hist  # array of integers
        self.sp_age_dict = sp_age_dict  # dict of species_id: age_array
        self.dist_matrix = dist_matrix  # 2D array
        self.disturbance = disturbance
        self.protection = protection
        self.pathogen = pathogen
        self.temperature = coord[0] * lat_steepness

    @property
    def n_individuals(self):
        return np.sum(self.species_hist)

    @property
    def available_space(self):
        return max(self.carrying_capacity - self.n_individuals, 0)

    @property
    def room_for_one(self):
        return min(self.available_space, 1)

    @property
    def n_species(self):
        return len(self.species_hist[self.species_hist > 0])
    
    @property  
    def shannon_div_idx(self):
        if self.n_individuals == 0:
            return 0
        pi = [count/self.n_individuals for count in self.species_hist if count != 0]
        ln_pi = np.log(pi)
        shannon = -sum(ln_pi*pi)
        return shannon
    


# -------------------------------------------------
# initialization functions
# -------------------------------------------------
def init_cell_objects(cell_id_n_coord, 
                      d_matrix, 
                      n_species, 
                      cell_carrying_capacity,
                      disturbance=0.0,
                      min_age = 2,
                      max_age = 10,
                      lat_steep=0.1,
                      max_age_sort=True):
    list_cells = []
    max_age_array = rnd_sp_max_age(n_species, min_age=min_age, max_age=max_age, sort=max_age_sort, seed=42)
    print("Max age array for each species:", max_age_array)
    sp_age_dict = {i: np.zeros(max_age_array[i], dtype=int) for i in range(n_species)}
    for i in cell_id_n_coord[:, 0]:
        c_coord = cell_id_n_coord[i, 1:]
        c_id = cell_id_n_coord[i, 0]
        c_species_hist = np.zeros(n_species, dtype=np.uint16)
        c_dist_matrix = d_matrix[i]
        c = CellClass(
            c_coord,
            c_id,
            c_dist_matrix,
            c_species_hist,
            copy.deepcopy(sp_age_dict),
            cell_carrying_capacity,
            disturbance=disturbance,
            lat_steepness=lat_steep,
        )
        list_cells.append(c)
    return list_cells


def init_propagate_species(
    curr_ind,
    max_n_ind,
    list_cells,
    cell_id_n_coord,
    cell_carrying_capacity,
    species_id,
    disp_rate=1.0,
    seed=42
):
    rng = np.random.default_rng(seed)
    aval_space = np.array([c.room_for_one for c in list_cells])
    # len(n_indviduals_per_cell) = total number of cells
    n_individuals_per_cell = np.array([c.n_individuals for c in list_cells])
    # len(n_individuals_per_cell_sp_i) = total number of cells
    n_individuals_per_cell_sp_i = np.zeros(len(list_cells))

    while True:
        # selectcurrentallocation cell
        for cell_id in curr_ind:
            aval_space[n_individuals_per_cell == cell_carrying_capacity] = 0
            c = list_cells[cell_id]
            disp = 1.0 / disp_rate
            disp_probability = disp * np.exp(-disp * c.dist_matrix)
            disp_vec = disp_probability.flatten()
            sampling_prob = disp_vec * aval_space

            # 如果没有canallocation空between，跳过
            if np.sum(sampling_prob) == 0:
                continue

            selected_cell = rng.choice(
                cell_id_n_coord[:, 0], p=sampling_prob / np.sum(sampling_prob)
            )
            # allocationindividual
            # update species histogram in selected cell
            # each iteration is dealing with only one species
            list_cells[selected_cell].species_hist[species_id] += 1
            # update age distribution in selected cell
            list_cells[selected_cell].sp_age_dict[species_id][0] += 1
            n_individuals_per_cell[selected_cell] += 1
            n_individuals_per_cell_sp_i[selected_cell] += 1
            curr_ind.append(selected_cell)

            if len(curr_ind) > max_n_ind:
                # n_individuals_per_cell_sp_i = np.unique(curr_ind,return_counts=True),
                # but includes 0s
                return (
                    n_individuals_per_cell,
                    n_individuals_per_cell_sp_i,
                )  # np.array(curr_ind),


def init_species_population(
    n_species,
    list_cells,
    cell_id_n_coord,
    cell_carrying_capacity,
    disp_rate=1.0,
    abundance_array=None,
    seed=42,
    verbose=1,
    half=False,
):
    """
    Initialize species population in the grid.
    If abundance_array is provided, use it as the total individuals per species.
    Otherwise, generate with default Weibull parameters.
    """
    rng = np.random.default_rng(seed)
    n_cells = len(list_cells)
    if half:
        n_total_individuals = cell_carrying_capacity * n_cells // 2
    else:
        n_total_individuals = cell_carrying_capacity * n_cells
    species_id_array = np.arange(n_species)
    if abundance_array is None:
        rel_frequencies = generate_weibull_abundance(n_species, weibull_k=0.75, scale_factor=1, seed=seed)
        _, abundance_array = sample_species_individuals(n_total_individuals, rel_frequencies, seed=seed)
    # n_individuals_per_species_full is the full count of individuals per species
    # it set the distribution limit and maximum of number of individual for each species during initiation
    rnd_species_order = rng.choice(species_id_array, n_species, replace=False)
    aval_space = np.array([c.room_for_one for c in list_cells])
    n_individuals_per_cell = np.array([c.n_individuals for c in list_cells])

    j = 1
    for species_id in rnd_species_order:
        max_n_ind = abundance_array[species_id]
        if verbose:
            print_update(
                    "%s/%s init species %s (%s ind.) %s"
                    % (j, n_species, species_id, max_n_ind, get_nature_emoji())
                )
        aval_space[n_individuals_per_cell == cell_carrying_capacity] = 0
        if np.sum(aval_space) == 0:
            print("\nNo available space left in any cell. Stopping initialization.")
            break
        start_cell = rng.choice(cell_id_n_coord[:, 0], p=aval_space/np.sum(aval_space))
        #start propogating the species from the start_cell
        curr_ind = [start_cell]
        #更新n_individuals_per_cellandn_individuals_added
        n_individuals_per_cell, n_individuals_added = init_propagate_species(
                                                                curr_ind,
                                                                max_n_ind,
                                                                list_cells,
                                                                cell_id_n_coord,
                                                                cell_carrying_capacity,
                                                                species_id,
                                                                disp_rate,
                                                                seed=seed
                                                            )
        j += 1
    return list_cells



# -------------------------------------------------
# Visualization Functions
# -------------------------------------------------
def plot_weibull_distribution(n_species, weibull_k=1.5, scale_factor=50, seed=None):
    """
    Plot the Weibull distribution for species abundance.
    """
    abundance = generate_weibull_abundance(n_species, weibull_k, scale_factor, seed)
    plt.figure(figsize=(8, 4))
    plt.hist(abundance, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Abundance")
    plt.ylabel("Number of Species")
    plt.title(f"Weibull Distribution (k={weibull_k}, scale={scale_factor})")
    plt.show()

def plot_species_distribution(list_cells, grid_size, species_id, title_prefix=""):
    abundance_grid = np.zeros((grid_size, grid_size))
    vmax = max([c.species_hist[species_id] for c in list_cells])
    vmin = min([c.species_hist[species_id] for c in list_cells])

    for c in list_cells:
        x, y = c.coord
        abundance_grid[int(x), int(y)] = c.species_hist[species_id]
    #standardize the abundance scale legend from 0 to carrying capacity
    plt.imshow(abundance_grid, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title(f"{title_prefix} Species {species_id} distribution")
    plt.colorbar(label="Abundance")
    plt.show()

def plot_disturbance_map(list_cells):
    """
    plot disturbance heatmap
    """
    disturbance = [c.disturbance for c in list_cells]
    grid_size = int(np.sqrt(len(list_cells)))
    disturbance = np.array(disturbance).reshape((grid_size, grid_size))
    plt.figure(figsize=(10, 6))
    plt.imshow(disturbance, cmap="viridis", origin="lower")
    plt.colorbar(label="Disturbance")
    plt.title("Disturbance Heat Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def plot_species_abundance_hist(list_cells):
    """
    plot每speciestotalindividual数直方图，检查是否符合 Weibull 分布
    """
    n_species = len(list_cells[0].species_hist)
    total_abundance = np.zeros(n_species)
    for cell in list_cells:
        total_abundance += cell.species_hist
    
    plt.figure(figsize=(8,4))
    plt.bar(range(n_species), total_abundance)
    plt.xlabel("Species ID")
    plt.ylabel("Total Individuals")
    plt.title("Species Total Abundance")
    plt.show()

def plot_biodiversity_map(list_cells):
    """
    plotspeciesdiversityheatmap
    """
    species_diversity = [c.shannon_div_idx for c in list_cells]
    grid_size = int(np.sqrt(len(list_cells)))
    species_diversity = np.array(species_diversity).reshape((grid_size, grid_size))
    plt.figure(figsize=(10, 6))
    plt.imshow(species_diversity, cmap="viridis", origin="lower")
    plt.colorbar(label="Shannon Index")
    plt.title("Species Diversity Heat Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def plot_n_species_map(list_cells):
    """
    plotspeciesquantityheatmap
    """
    n_species = [c.n_species for c in list_cells]
    grid_size = int(np.sqrt(len(list_cells)))
    species_diversity = np.array(species_diversity).reshape((grid_size, grid_size))
    plt.figure(figsize=(10, 6))
    plt.imshow(n_species, cmap="viridis", origin="lower")
    plt.colorbar(label="Number of Species")
    plt.title("Species Diversity Heat Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()
# -------------------------------------------------
# load pickle file
# -------------------------------------------------
def load_pickle_file(pkl):
    with open(pkl, "rb") as pkl:
        list_cells = pickle.load(pkl)
        return list_cells


# -------------------------------------------------
# Demo
# -------------------------------------------------
# if __name__ == "__main__":

    # plot_weibull_distribution(n_species=25, weibull_k=0.75, scale_factor=1, seed=42)

    # grid_size = 50
    # n_species = 25
    # carrying_capacity = 25

    # d_matrix = get_all_to_all_dist(grid_size)
    # cell_id_n_coord = get_coordinates(grid_size)
    # list_cells = init_cell_objects(cell_id_n_coord, d_matrix, n_species, carrying_capacity, lat_steep=0.1)

    # init_species_population(
    #     n_species=n_species,
    #     list_cells=list_cells,
    #     cell_id_n_coord=cell_id_n_coord,
    #     cell_carrying_capacity=carrying_capacity,
    #     disp_rate=1.0,
    # )

    # for sp_id in range(n_species):
    #     plot_species_distribution(list_cells, grid_size, sp_id)


