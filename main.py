import numpy as np
import GeneticAlgorithmToolbox  # wind farm layout optimization using genetic algorithms classes
import os


def main():
    # parameters for the genetic algorithm
    ELITEPB = 0.2
    TOURN_SIZE = 5
    CXPB = 0.8
    MUTPB = 0.1
    SLXPB = 0.5

    # wind farm size, cells
    COLS = 10
    ROWS = 15
    CELL_WIDTH = 10
    CELL_HEIGHT = 10
    D = 10

    N = 40  # number of wind turbines
    POP_SIZE = 200  # population size, number of individuals in a population
    GENERATION = 150  # number of genetic algorithm iterations

    results_data_folder = "data/results"
    if not os.path.exists(results_data_folder):
        os.makedirs(results_data_folder)

    # create an object of WindFarmGenetic
    wfg = Toolbox2D.FarmOptimizaiton(rows=ROWS, cols=COLS, N=N, pop_size=POP_SIZE, iteration=GENERATION, cell_width=CELL_WIDTH,
                              cell_height=CELL_HEIGHT, elite_rate=ELITEPB,
                              cross_rate=CXPB, random_rate=SLXPB, tourn_size=TOURN_SIZE,
                              mutate_rate=MUTPB, D=D)

    wfg.current_speed()

    # n_init_pops : number of initial populations
    init_pops_data_folder = "data/init_pops"
    if not os.path.exists(init_pops_data_folder):
        os.makedirs(init_pops_data_folder)

    n_init_pops = 1
    for i in range(n_init_pops):
        wfg.initial_population("{}/init_{}.dat".format(init_pops_data_folder, i))

    n_run_times = 1  # number of run times
    # result_arr stores the best conversion efficiency of each run
    result_arr = np.zeros((n_run_times, 2), dtype=np.float32)
    eta_runs = np.zeros((GENERATION, n_run_times), dtype=np.float32)
    layout_runs = np.zeros((n_run_times, COLS * ROWS), dtype=np.float32)
    if not os.path.exists(results_data_folder):
        os.makedirs(results_data_folder)
    for i in range(0, n_run_times):  # run times
        print("Run {}:".format(i + 1))
        wfg.load_init_pop("{}/init_{}.dat".format(init_pops_data_folder, i))
        eta, eta_gen, best_layout = wfg.genetic_algorithm()
        result_arr[i] = eta
        eta_runs[:, i] = eta_gen
        layout_runs[i, :] = best_layout
    filename = "{}/best_eta_{}.dat".format(results_data_folder, N)
    np.savetxt(filename, result_arr, fmt='%f', delimiter="  ")
    filename = "{}/eta_{}.dat".format(results_data_folder, N)
    np.savetxt(filename, eta_runs, fmt='%f', delimiter="  ")

    s_index = np.argsort(-result_arr[:, 0], axis=0)
    best_layout_runs = layout_runs[s_index[0].item()]
    filename = "{}/best_layouts_{}.dat".format(results_data_folder, N)
    np.savetxt(filename, best_layout_runs, fmt='%2f', delimiter="  ")


if __name__ == '__main__':
    main()
