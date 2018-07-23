import numpy as np
import pygmo as pg
import os
import joblib
from scipy.constants import g as gravity
from tqdm import tqdm

from systools import Simulator, TrajectoryProblem, PendelWagenSystem
from functions import spline_func_gen

# -----------------------------------------
# Vorgabe der Wagenposition
# -----------------------------------------
func = spline_func_gen(kind=5, diff=2, bc_type=[[(1, 0), (2, 0)], [(1, 0), (2, 0)]])

parameter_values = dict(g=gravity, m1=1, m2=0.1, l=0.5)
sys = PendelWagenSystem().get_sim_model(parameter_values)

x0 = [0,0,0,0]
t0 = 0.0
T = 5.0

sim = Simulator(sys, x0, t0, T, input_func=func, func_parameters={'n': 2, 'start_end': (0.0, 0.5)})
prob = pg.problem(TrajectoryProblem(sim, weights=(1,0,1,0), bounds=[(-1, 1), (-1, 1)]))


positions = np.linspace(0.1, 0.9, 5)
#positions = [0.5]

for target_position in positions:

    sim.func_parameters.update(start_end=(0.0, target_position))

    k = 400
    bounds = [-2, 2, -2, 2]
    P1, P2 = np.meshgrid(np.linspace(bounds[0], bounds[1], k), np.linspace(bounds[2], bounds[3], k))
    P = np.vstack((P1.flatten(), P2.flatten())).T

    def get_results(params, output, i):
        res = sim.solve(params)
        output[i, :] = res.T[:, -1]

    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    output_filename_memmap = os.path.join(folder, 'output_memmap')
    X = np.memmap(output_filename_memmap, shape=(k**2, 4), dtype='float32', mode='w+')

    joblib.Parallel(n_jobs=8)(joblib.delayed(get_results)(p, X, i) for i, p in tqdm(enumerate(P)))

    out = [P1, P2, X, k, bounds]
    np.save(f"PWS_h_{target_position}_{bounds[0]}--{bounds[1]}x{bounds[2]}--{bounds[3]}_{k}.npy", out)