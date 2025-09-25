'''
not used currently
This script concludes the characteristics of species, including
- birth rate
- mortality rate
- carrying capacity
- diffusion rate
- sensitivity to disturbance
'''

class Species:
    def __init__(self, name, birth_rate, mortality_rate, carrying_capacity, diffusion_rate,
                 sensitivity_D=1.0, sensitivity_B=1.0, sensitivity_K=1.0):
        self.name = name
        self.abundance = 0
        self.birth_rate = birth_rate  # intrinsic birth rate
        self.mortality_rate = mortality_rate  # intrinsic mortality rate
        self.carrying_capacity = carrying_capacity  # intrinsic carrying capacity
        self.diffusion_rate = diffusion_rate  # diffusion rate on the map
        self.sensitivity_D = sensitivity_D  # sensitivity to D disturbance
        self.sensitivity_B = sensitivity_B  # sensitivity to B disturbance
        self.sensitivity_K = sensitivity_K  # sensitivity to K disturbance

    def __repr__(self):
        return f"Species({self.name}, b={self.birth_rate}, m={self.mortality_rate}, K={self.carrying_capacity}, d={self.diffusion_rate})"