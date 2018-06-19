import numpy as np
import sympy as sp
import os
import joblib
from scipy.constants import g as gravity
import symbtools as st
import symbtools.modeltools as mt
from tqdm import tqdm

from sample_gen import SampleGenerator
from functions import sidestepping_spline_func_gen

# -----------------------------------------
# Pendel-Wagen System mit hängendem Pendel
# -----------------------------------------

t = sp.Symbol('t')
n_p = 1
n_q = 1
n = n_p + n_q
pp = st.symb_vector(("varphi", ))
qq = st.symb_vector(("q", ))

ttheta = st.row_stack(pp, qq)
st.make_global(ttheta)

params = sp.symbols('m1, m2, l, g, q_r, t, T')
st.make_global(params)

ex = sp.Matrix([1,0])
ey = sp.Matrix([0,1])

# Koordinaten der Schwerpunkte und Gelenke
S1 = ex * q  # Schwerpunkt Wagen
G2 = S1      # Pendel-Gelenk

# Schwerpunkt des Pendels (Pendel zeigt für kleine Winkel nach oben)
S2 = G2 + mt.Rz(varphi)*-ey*l

# Zeitableitungen der Schwerpunktskoordinaten
S1d, S2d  = st.col_split(st.time_deriv(st.col_stack(S1, S2), ttheta))

# Energie
E_rot = 0 # (Punktmassenmodell)
E_trans = (m1*S1d.T*S1d  +  m2*S2d.T*S2d) / 2

E = E_rot + E_trans[0]

V = m2*g*S2[1]

# Partiell linearisiertes Model
mod = mt.generate_symbolic_model(E, V, ttheta, [0, sp.Symbol("u")])
mod.calc_state_eq()
mod.calc_coll_part_lin_state_eq()
f = mod.ff
G = mod.gg

# Model- und Simulationsparameter
parameter_values = [(g, gravity), (m1, 1), (m2, 0.1), (l, 0.5)]
x0 = [0.0, 0.0, 0.0, 0.0]
t0 = 0.0
T = 5.0

# -----------------------------------------
# Vorgabe der Wagenposition
# -----------------------------------------

target_position = 0.5

func = sidestepping_spline_func_gen(kind=5, diff=2, start_end=(0.0, target_position),
                                    bc_type=[[(1, 0), (2, 0)], [(1, 0), (2, 0)]])

gen = SampleGenerator(mod.ff, mod.gg, mod.xx, x0, t0, T,
                      input_func=func, model_parameters=parameter_values,
                      func_parameters={'n': 2})

k = 400
bounds = [-5, 10, -5, 10]
P1, P2 = np.meshgrid(np.linspace(bounds[0], bounds[1], k), np.linspace(bounds[2], bounds[3], k))
P = np.vstack((P1.flatten(), P2.flatten())).T


def get_results(params, output, i):
    res = gen.sim(params)
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