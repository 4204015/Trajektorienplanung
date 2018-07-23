import sympy as sp
import time
import symbtools.modeltools as mt
import matplotlib.pyplot as plt

from functions import sidestepping_spline_func_gen
from visualisation import PendelWagenSystem as PWSVis

import numpy as np
import pandas as pd
import scipy.integrate as sci
import symbtools as st

from sklearn.base import BaseEstimator
from bokeh.plotting import show
from bokeh.layouts import row, gridplot
from bokeh.io import push_notebook
from bokeh.models import Band, ColumnDataSource
from bokeh.models import Range1d


class Simulator(BaseEstimator):
    def __init__(self, model, x0, t0, T=None, input_func_gen=None, func_parameters=None,
                 step_size=0.1e-3, solver='odeint'):

        self.model = model
        self.input_func_gen = input_func_gen
        self.func_parameters = func_parameters
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.step_size = step_size
        self.solver = solver

    def solve(self, param, T=None, additional_time=0, use_sp2c=True, start_end=None, input_func=None):
        assert bool(self.T) != bool(T)  # xor

        T_spline = self.T if not T else T
        T_sim = T_spline + additional_time
        tt = np.arange(self.t0, self.t0 + T_sim, self.step_size)

        if not input_func:
            def input_func(t):
                return self.input_func_gen(t=t, t_range=[self.t0, self.t0 + T_spline], param=param,
                                           start_end=self.func_parameters.get(
                                               'start_end', (start_end if start_end else (0.0, 0.0))))

        sim_func = self.model.create_simfunction(input_function=input_func,
                                                 use_sp2c=use_sp2c)

        if self.solver == 'odeint':
            res = sci.odeint(func=sim_func, y0=self.x0, t=tt, rtol=1e-6)
        else:

            res = None # TODO

        return res, tt

    def plot_input_func(self, param, start_end=None):
        tt = np.arange(self.t0, self.t0 + self.T, self.step_size)

        def input_func(t):
            return self.input_func_gen(t=t, t_range=[self.t0, self.t0 + self.T], param=param,
                                       start_end=self.func_parameters.get('start_end', start_end))

        u = [input_func(t_) for t_ in tt]

        fig = plt.figure()
        plt.plot(tt, u)
        plt.xlabel("t")
        plt.ylabel("u(t)")
        plt.grid()
        plt.tight_layout()


class LocalOptimizer:
    def __init__(self, simulator):
        self.simulator = simulator

    def _esimate_jacobian(self, target, dx=0.001):
        Fa = np.array(self.simulator.solve(self.simulator.param)[-1, :]) - target
        J = np.zeros((len(self.simulator.x0), len(self.simulator.param)))

        for i in range(len(self.simulator.param)):
            param = self.simulator.param
            param[i] = param[i] + dx

            Fb = np.array(self.simulator.solve(param)[-1, :]) - target
            J[:, i] = np.gradient([Fa, Fb], dx, axis=0)[0]

        return J, Fa

    def search(self, target, search_algorithm='gd', stopping=1000,
               learning_rate=0.001, weights=None, alpha=0.9, beta1=0.9, beta2=0.999,
               epsilon=10e-8, plotter=None):

        assert len(target) == len(self.simulator.param)

        # inital guess for all parameters
        self.simulator.param = self.simulator.func_parameters.get(
            'initial')  # , np.hstack((np.random.uniform(*span) for span in self.func_parameters['range'])))

        params = np.array(self.simulator.param)
        objective_function = []

        if weights:
            W = np.diag(weights)
        else:
            W = np.identity(n=len(self.simulator.param))

        v = 0
        m = 0

        while len(objective_function) < stopping:
            t = len(objective_function) + 1
            J, F = self._esimate_jacobian(target)

            gradient = J.T @ W @ F

            objective = F.T @ W @ F
            objective_function.append(objective)

            if objective < 0.0001:# or objective > 2*objective_function[0]:
                msg = "Stopping because convergence" if objective < 0.0001 else "explosion of loss"
                print(msg + f": {objective}")
                plotter.update({'loss': objective, 'param': self.simulator.param, 'phase_space': F})
                break

            if search_algorithm == 'momentum':
                v = v * alpha + (1 - alpha) * gradient
                self.simulator.param -= learning_rate * v

            elif search_algorithm == 'adam':
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient) ** 2

                m_ = m / (1 - beta1 ** t)
                v_ = v / (1 - beta2 ** t)

                self.simulator.param -= learning_rate * m_ / (np.sqrt(v_) + epsilon)
            else:
                self.simulator.param -= learning_rate * gradient

            if plotter:
                plotter.update({'loss': objective, 'param': self.simulator.param, 'phase_space': F})

            params = np.vstack((params, self.simulator.param))

        return params, objective_function


class TrainingPlotter:
    def __init__(self, *args, types):
        self.figures = args
        self.types = types
        assert len(self.figures) == len(types)
        self.plots = [getattr(fig, plot_type)([0], [0]) for fig, plot_type in zip(args, types)]

        if len(self.figures) > 3:
            f = list(self.figures) + [None]*(6-len(self.figures))
            grid = gridplot(np.reshape(f, (2, 3)).tolist())
            self.plot_handle = show(grid, notebook_handle=True)

        else:
            self.plot_handle = show(row(*self.figures), notebook_handle=True)

        self.i = 0

    def update(self, data_dict):
        for plot, fig in zip(self.plots, self.figures):

            try:
                try:
                    key, idx = fig.title.text.split('-')
                    data = data_dict[key][int(idx)]
                except ValueError:
                    data = data_dict[fig.title.text]
            except KeyError:
                if self.i == 0:
                    pass
                else:
                    continue

            new_data = dict()
            if self.i == 0:

                if fig.title.text == 'phase_space':
                    fig.xaxis.axis_label = 'phi'
                    fig.yaxis.axis_label = 'phi_dot'
                    new_data['x'], new_data['y'], *_ = [[data[0]], [data[2]]]

                elif fig.title.text == 'model_ranges':

                    new_data = {'lower': [1.0], 'upper': [1.0], 'x': [self.i], 'y': [1.0]}
                    plot.data_source.data.update(new_data)

                    self.band = Band(base='x', lower='lower', upper='upper', source=plot.data_source, level='underlay',
                                     fill_alpha=1.0, line_width=1, line_color='black')
                    fig.add_layout(self.band)
                    fig.xaxis.axis_label = 'iteration'
                else:
                    fig.xaxis.axis_label = 'iteration'

                    new_data['x'] = [self.i]
                    new_data['y'] = [data]


            else:
                if fig.title.text == 'phase_space':
                    new_data['x'] = plot.data_source.data['x'] + [data[0]]
                    new_data['y'] = plot.data_source.data['y'] + [data[2]]
                elif fig.title.text == 'model_ranges':
                    new_data['x'] = plot.data_source.data['x'] + [self.i]
                    new_data['y'] = plot.data_source.data['y'] + [data['y']]
                    new_data['lower'] = plot.data_source.data['lower'] + [data['lower']]
                    new_data['upper'] = plot.data_source.data['upper'] + [data['upper']]
                    fig.y_range = Range1d(np.min(plot.data_source.data['lower']), 2)
                else:
                    new_data['x'] = plot.data_source.data['x'] + [self.i]
                    new_data['y'] = plot.data_source.data['y'] + [data]


            plot.data_source.data.update(new_data)
        self.i += 1

        push_notebook(handle=self.plot_handle)

    def reset(self):
        self.i = 0

    def draw(self):
        pass


class PendelWagenSystem:
    def __init__(self, inverted=True, calc_coll_part_lin=True):

        self.inverted = inverted
        self.calc_coll_part_lin = calc_coll_part_lin
        # -----------------------------------------
        # Pendel-Wagen System mit hängendem Pendel
        # -----------------------------------------

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

        if inverted:
            S2 = G2 + mt.Rz(varphi) * ey * l

        else:
            S2 = G2 + mt.Rz(varphi) * -ey * l

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

        self.mod = mod

    def get_sim_model(self, model_parameters):
        if self.calc_coll_part_lin:
            return st.SimulationModel(self.mod.ff, self.mod.gg, self.mod.xx, model_parameters)
        else:
            return st.SimulationModel(self.mod.f, self.mod.g, self.mod.xx, model_parameters)

    def get_animation_model(self, l, car_width=0.1, car_height=0.05, pendulum_size=0.05):
        rod_length = -l if not self.inverted else l
        return PWSVis(rod_length=rod_length, car_width=car_width, car_height=car_height, pendulum_size=pendulum_size)


def logistic(x, L=1.0, k=1.0, x0=0.0):
    return L / (1.0 + np.exp(-k * (x-x0)))


class TrajectoryProblem:
    def __init__(self, simulator, bounds, weights=(1, 1, 1, 1), time_as_param=False):
        self.sim = simulator
        self.W = np.diag(weights)
        self.bounds = bounds # e.g. [(-5, 5), (-10, 10)]
        self.time_as_param = time_as_param

    def fitness(self, x):
        if self.time_as_param:
            T = x[-1]
        else:
            T = 0

        res = self.sim.solve(param=x[0:2], T=T)
        phi = res[0][:, 0]
        final_state = res[0].T[:, -1]

        # Root Mean Squared Error of the weighted states
        rms = np.sqrt(final_state.T @ self.W @ final_state)

        # penalty for big angles
        phi_penalty = logistic(x=np.max(np.abs(phi)), L=80, k=8.5, x0=np.pi*0.65)

        # penalty for big params
        #factor = 0.0
        #param_penalty = np.sqrt(x[0:2].T @ x[0:2])

        return [rms + phi_penalty]# + factor*param_penalty]

    def get_bounds(self):
        return tuple(zip(*self.bounds))

