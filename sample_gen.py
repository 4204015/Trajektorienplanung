import numpy as np
import tqdm
import sympy as sp
import scipy.integrate as sci
import numba as nb
import symbtools as st
import symbtools.modeltools as stm


class SampleGenerator:
    def __init__(self, f, g, x, x0, t0, T, model_parameters, input_func, func_parameters=None, seed=True, search_space=None):
        self.system = st.SimulationModel(f, g, x, model_parameters)
        self.input_func = input_func
        self.func_parameters = func_parameters
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.t = np.arange(self.t0, self.t0 + self.T, 0.1e-3)
        self.search_space = search_space

        if seed:
            np.random.seed(42)

    def sim(self, param):

        #func = sp.lambdify(args=sp.Symbol("t"),
        #                   expr=self.input_func,
        #                   modules=['numpy'])

        #sim_func=lambda t, x: self.system.create_simfunction(input_function=func)(x, t)

        func = lambda t: self.input_func(t, [self.t0, self.t0 + self.T], param)

        #sim_func = lambda x, t: self.system.create_simfunction(input_function=func)(x, t)

        # Solve an initial value problem for self.system
        #res = sci.solve_ivp(fun=sim_func,
        #                    t_span=[self.t0, self.t0 + self.T],
         #                   y0=self.x0,
         #                   method='RK45',
         #                   vectorized=False)

        res = sci.odeint(func=self.system.create_simfunction(input_function=func),
                         y0=self.x0,
                         t=self.t)

        return res
    
    def generate(self, n=10):
        x = np.zeros((n, len(self.x0)))
        y = np.zeros((n, self.func_parameters['n']))
        pbar = tqdm.tqdm_notebook(total=n)            

        i = 0
        while i < n:
            param = np.hstack((np.random.uniform(*span) for span in self.func_parameters['range']))
            res = self.sim(param)
            x1 = res[-1, :]

            if not self.inside(x1):
                continue

            x[i, :] = x1
            y[i, :] = param
            i += 1
            pbar.update(1)

        pbar.close()

        return x, y, res
    
    def inside(self, x1):
        if self.search_space:
            return all(self.search_space[i][0] < x < self.search_space[i][1] for i, x in enumerate(x1))
        else:
            return True


