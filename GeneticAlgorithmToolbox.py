import math
import numpy as np
import time
import random
import concurrent.futures

__version__ = "1.0.0"


class FarmOptimizaiton():
    b = 1.32
    rho = 1024
    e1 = 0.12182
    e2 = 0.33578
    v_rated = 1.7
    v_cut_in = 0.34
    v_cut_out = 2.5

    # constructor of class WindFarmGenetic
    def __init__(self, rows=0, cols=0, N=0, pop_size=0, iteration=0, cell_width=0, cell_height=0, elite_rate=0.2,
                 cross_rate=0.6, random_rate=0.5, tourn_size=3, mutate_rate=0.1, D=0.1):
        self.f_theta_v = None
        self.velocity = None
        self.theta = None
        self.rows = rows
        self.cols = cols
        self.N = N
        self.pop_size = pop_size
        self.iteration = iteration
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cell_width_half = cell_width * 0.5
        self.random_rate = random_rate
        self.elite_rate = elite_rate
        self.cross_rate = cross_rate
        self.tourn_size = tourn_size
        self.mutate_rate = mutate_rate
        self.init_pop = None
        self.init_pop_nonezero_indices = None
        self.D = D

    def current_speed(self):
        self.theta = np.array([0], dtype=np.float32)
        self.velocity = np.array([1.2225], dtype=np.float32)
        self.f_theta_v = np.array([[1.0]], dtype=np.float32)
        return

    def initial_population(self, fname):
        np.random.seed(seed=int(time.time()))
        layouts = np.zeros((self.pop_size, self.rows * self.cols), dtype=np.int32)
        positionX = np.random.randint(0, self.cols, size=(self.N * self.pop_size * 2))
        positionY = np.random.randint(0, self.rows, size=(self.N * self.pop_size * 2))
        ind_rows = 0
        ind_pos = 0
        while ind_rows < self.pop_size:
            layouts[ind_rows, positionX[ind_pos] + positionY[ind_pos] * self.cols] = 1
            if np.sum(layouts[ind_rows, :]) == self.N:
                ind_rows += 1
            ind_pos += 1
            if ind_pos >= self.N * self.pop_size * 2:
                print("Not enough positions")
                break
        self.init_pop = layouts
        np.savetxt(fname, self.init_pop, fmt='%0.2f', delimiter="  ")
        return

    def load_init_pop(self, fname):
        self.init_pop = np.genfromtxt(fname, delimiter="  ", dtype=np.float32)
        self.init_pop_nonezero_indices = np.zeros((self.pop_size, self.N), dtype=np.int32)
        for ind_init_pop in range(self.pop_size):
            ind_indices = 0
            for ind in range(self.rows * self.cols):
                if self.init_pop[ind_init_pop, ind] == 1:
                    self.init_pop_nonezero_indices[ind_init_pop, ind_indices] = ind
                    ind_indices += 1
        return

    def power_output(self, vel, D1):
        power = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            if vel[i] < self.v_cut_in:
                p_t = 0
            elif vel[i] < self.v_rated:
                p_t = (4 * self.e1 * (1 - self.e1) ** 2) * 0.5 * self.rho * np.pi * (D1 / 2) ** 2 * vel[i] ** 3 \
                      + (4 * (self.e2 - 2 * self.e1) * (1 - self.e2) ** 2) * 0.5 * self.rho * np.pi * \
                      (self.b * (D1 / 2)) ** 2 * vel[i] ** 3
            elif vel[i] < self.v_cut_out:
                p_t = (4 * self.e1 * (1 - self.e1) ** 2) * 0.5 * self.rho * np.pi * (D1 / 2) ** 2 * self.v_rated ** 3 \
                      + (4 * (self.e2 - 2 * self.e1) * (1 - self.e2) ** 2) * 0.5 * self.rho * np.pi * \
                      (self.b * (D1 / 2)) ** 2 * self.v_rated ** 3
            else:
                p_t = 0

            power[i] = p_t
        return power

    def wake_effect(self, trans_xyz_position):
        sorted_index = np.argsort(-trans_xyz_position[1, :])  # y value descending
        wake_deficiency = np.zeros(self.N, dtype=np.float32)
        wake_deficiency[sorted_index[0]] = 0
        ####################################
        D1 = self.D
        D2 = D1 * self.b
        e = np.array([self.e1, self.e2])
        D_x0 = np.sqrt((1 - e[0]) / (1 - 2 * e[0]) * D1 ** 2 - 2 * (2 * e[0] - e[1]) * (1 - e[1]) / (
                (1 - 2 * e[0]) * (1 - 2 * e[1] + 2 * e[0])) * D2 ** 2)
        d2 = D2 * np.sqrt((1 - e[1]) / (1 - 2 * e[1] + 2 * e[0]))
        x0 = 3.1699999 * D1
        ec = (D_x0 - D1) / (2 * x0)
        ####################################
        for i in range(1, self.N):
            for j in range(i):
                dx = np.absolute(trans_xyz_position[0, sorted_index[i]] - trans_xyz_position[0, sorted_index[j]])
                dy = np.absolute(trans_xyz_position[1, sorted_index[i]] - trans_xyz_position[1, sorted_index[j]])
                r = D1 / 2
                if dy == 0:
                    d = 0
                else:
                    R = r + ec * dy  # R = wake radius; r = turbine radius
                    if dx >= r + R:
                        inter_area = 0
                    elif dx >= np.sqrt(R ** 2 - r ** 2):
                        alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
                        beta = np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
                        A1 = alpha * R ** 2
                        A2 = beta * r ** 2
                        A3 = R * dx * np.sin(alpha)
                        inter_area = A1 + A2 - A3
                    elif dx >= R - r:
                        alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
                        beta = np.pi - np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
                        A1 = alpha * R ** 2
                        A2 = beta * r ** 2
                        A3 = R * dx * np.sin(alpha)
                        inter_area = np.pi * r ** 2 - (A2 + A3 - A1)
                    else:
                        inter_area = np.pi * r ** 2

                    if dy < x0:
                        Uij = self.velocity * (
                                ((1 - e[1]) * D2 ** 2 + (D_x0 ** 2 - d2 ** 2) * (1 - 2 * e[0])) / (
                                D1 + 2 * ec * dy) ** 2)
                    else:
                        Uij = (self.velocity * (1 + 2 * e[0] - 2 * e[1]) * d2 ** 2 +
                               (D_x0 ** 2 - d2 ** 2) * self.velocity * (1 - 2 * e[0]) +
                               self.velocity * ((D1 + 2 * ec * dy) ** 2 - (D1 + 2 * ec * x0) ** 2)) / (
                                      D1 + 2 * ec * dy) ** 2
                    d = np.sqrt(inter_area / (np.pi * r ** 2)) * (self.velocity - Uij)

                wake_deficiency[sorted_index[i]] += d ** 2

            wake_deficiency[sorted_index[i]] = np.sqrt(wake_deficiency[sorted_index[i]])

        return wake_deficiency

    def genetic_algorithm(self):
        fitness_generations = np.zeros(self.iteration, dtype=np.float32)  # best fitness value in each generation
        best_layout_generations = np.zeros((self.iteration, self.rows * self.cols),
                                           dtype=np.float32)  # best layout in each generation

        pop = np.copy(self.init_pop)
        pop_indices = np.copy(self.init_pop_nonezero_indices)  # each row is a layout cell indices.

        for gen in range(self.iteration):
            print("Generation {}:".format(gen))
            with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
                results = executor.map(self.fitness, pop)
            fitness_value = np.array(list(results))
            sorted_index = np.argsort(-fitness_value)  # fitness value descending from largest to least
            pop = pop[sorted_index, :]
            pop_indices = pop_indices[sorted_index, :]
            if gen == 0:
                fitness_generations[gen] = fitness_value[sorted_index[0]]
                best_layout_generations[gen, :] = pop[0, :]
            else:
                if fitness_value[sorted_index[0]] > fitness_generations[gen - 1]:
                    fitness_generations[gen] = fitness_value[sorted_index[0]]
                    best_layout_generations[gen, :] = pop[0, :]
                else:
                    fitness_generations[gen] = fitness_generations[gen - 1]
                    best_layout_generations[gen, :] = best_layout_generations[gen - 1, :]
            n_parents, parent_layouts, parent_pop_indices = self.selection(pop=pop, pop_indices=pop_indices)
            self.crossover(pop=pop, pop_indices=pop_indices, n_parents=n_parents, parent_layouts=parent_layouts,
                           parent_pop_indices=parent_pop_indices)
            self.mutation(pop=pop, pop_indices=pop_indices)
            print(fitness_generations[gen])

        eta_generations = np.copy(fitness_generations)
        return eta_generations[self.iteration - 1], eta_generations, best_layout_generations[-1, :]

    def mutation(self, pop, pop_indices):
        np.random.seed(seed=int(time.time()))
        for i in range(self.pop_size):
            if np.random.randn() > self.mutate_rate:
                continue
            while True:
                turbine_pos = np.random.randint(0, self.cols * self.rows)
                if pop[i, turbine_pos] == 1:
                    break
            while True:
                null_turbine_pos = np.random.randint(0, self.cols * self.rows)
                if pop[i, null_turbine_pos] == 0:
                    break
            pop[i, turbine_pos] = 0
            pop[i, null_turbine_pos] = 1
            for j in range(self.N):
                if pop_indices[i, j] == turbine_pos:
                    pop_indices[i, j] = null_turbine_pos
                    break
            pop_indices[i, :] = np.sort(pop_indices[i, :])
        return

    def crossover(self, pop, pop_indices, n_parents, parent_layouts, parent_pop_indices):
        n_counter = 0
        np.random.seed(seed=int(time.time()))  # init random seed
        while n_counter < self.pop_size:
            if np.random.randn() < self.cross_rate:
                male = np.random.randint(0, n_parents)
                female = np.random.randint(0, n_parents)
                if male != female:
                    cross_point = np.sort(np.random.randint(1, self.N, 2))
                    if cross_point[0] != cross_point[1] and parent_pop_indices[male, cross_point[0] - 1] < parent_pop_indices[female, cross_point[0]] and parent_pop_indices[female, cross_point[1] - 1] < parent_pop_indices[male, cross_point[1]]:
                        pop[n_counter, :] = 0
                        pop[n_counter, :parent_pop_indices[male, cross_point[0] - 1] + 1] = parent_layouts[male, :parent_pop_indices[male, cross_point[0] - 1] + 1]
                        pop[n_counter, parent_pop_indices[female, cross_point[0]]: parent_pop_indices[female, cross_point[1] - 1] + 1] = parent_layouts[female, parent_pop_indices[female, cross_point[0]]:parent_pop_indices[female, cross_point[1] - 1] + 1]
                        pop[n_counter, parent_pop_indices[male, cross_point[1]]:] = parent_layouts[male, parent_pop_indices[male, cross_point[1]]:]

                        pop_indices[n_counter, :cross_point[0]] = parent_pop_indices[male, :cross_point[0]]
                        pop_indices[n_counter, cross_point[0]:cross_point[1]] = parent_pop_indices[female, cross_point[0]:cross_point[1]]
                        pop_indices[n_counter, cross_point[1]:] = parent_pop_indices[male, cross_point[1]:]

                    else:
                        cross_point = np.random.randint(1, self.N)
                        if parent_pop_indices[male, cross_point - 1] < parent_pop_indices[female, cross_point]:
                            # preventing position repetition
                            pop[n_counter, :] = 0
                            pop[n_counter, :parent_pop_indices[male, cross_point - 1] + 1] = parent_layouts[male, :parent_pop_indices[male, cross_point - 1] + 1]
                            pop[n_counter, parent_pop_indices[female, cross_point]:] = parent_layouts[female, parent_pop_indices[female, cross_point]:]

                            pop_indices[n_counter, :cross_point] = parent_pop_indices[male, :cross_point]
                            pop_indices[n_counter, cross_point:] = parent_pop_indices[female, cross_point:]
            n_counter += 1
        return

    def selection(self, pop, pop_indices):
        n_elite = int(self.pop_size * self.elite_rate)
        parents_ind = [i for i in range(n_elite)]
        np.random.seed(seed=int(time.time()))  # init random seed
        for _ in range(self.pop_size - n_elite):
            selection_ix = np.random.randint(n_elite, self.pop_size)
            for ix in np.random.randint(n_elite, self.pop_size, self.tourn_size - 1):
                if ix < selection_ix:
                    selection_ix = ix
            if np.random.randn() < self.random_rate:
                parents_ind.append(selection_ix)
        parent_layouts = pop[parents_ind, :]
        parent_pop_indices = pop_indices[parents_ind, :]
        return len(parent_pop_indices), parent_layouts, parent_pop_indices

    def fitness(self, individual):
        xy_position = np.zeros((2, self.N), dtype=np.float32)  # x y position
        cr_position = np.zeros((2, self.N), dtype=np.int32)  # column row position
        ind_pos = 0
        for ind in range(self.rows * self.cols):
            if individual[ind] == 1:
                r_i = np.floor(ind / self.cols)
                c_i = np.floor(ind - r_i * self.cols)
                cr_position[0, ind_pos] = c_i
                cr_position[1, ind_pos] = r_i
                xy_position[0, ind_pos] = c_i * self.cell_width + self.cell_width_half
                xy_position[1, ind_pos] = r_i * self.cell_height + 0.5 * self.cell_height
                ind_pos += 1
        speed_deficiency = self.wake_effect(xy_position)
        actual_velocity = self.velocity - speed_deficiency
        n_f = round(np.sum(self.power_output(actual_velocity, self.D)) / np.sum(
                        self.power_output(np.repeat(self.velocity, self.N), self.D)), 3)
        return n_f

