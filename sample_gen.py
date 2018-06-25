import numpy as np
import tqdm
import scipy.integrate as sci
import symbtools as st

from bokeh.plotting import show
from bokeh.layouts import row, gridplot
from bokeh.io import push_notebook


class SampleGenerator:
    def __init__(self, f, g, xx, x0, t0, T, input_func=None,
                 model_parameters=(), func_parameters=None,
                 search_space=None, step_size=0.1e-3, seed=42):
        self.system = st.SimulationModel(f, g, xx, model_parameters)
        self.input_func = input_func
        self.func_parameters = func_parameters
        self.x0 = x0
        self.t0 = t0
        self.T = T
        self.t = np.arange(self.t0, self.t0 + self.T, step_size)
        self.search_space = search_space

        np.random.seed(seed)

    def set_input_func(self, input_func, func_parameters):
        self.input_func = input_func
        self.func_parameters = func_parameters

    def _get_jacobian(self, target, dx=0.001):
        Fa = np.array(self.sim(self.param)[-1, :]) - target
        J = np.zeros((len(self.x0), len(self.param)))

        for i in range(len(self.param)):
            param = self.param
            param[i] = param[i] + dx

            Fb = np.array(self.sim(param)[-1, :]) - target
            J[:, i] = np.gradient([Fa, Fb], dx, axis=0)[0]

        return J, Fa

    def search(self, target, search_algorithm='gd', stopping=1000,
               learning_rate=0.001, weights=None, alpha=0.9, beta1=0.9, beta2=0.999,
               epsilon=10e-8, plotter=None):

        assert len(target) == len(self.param)

        # inital guess for all parameters
        self.param = self.func_parameters.get(
            'initial')  # , np.hstack((np.random.uniform(*span) for span in self.func_parameters['range'])))

        params = np.array(self.param)
        objective_function = []

        if weights:
            W = np.diag(weights)
        else:
            W = np.identity(n=len(self.param))

        v = 0
        m = 0

        while len(objective_function) < stopping:
            t = len(objective_function) + 1
            J, F = self._get_jacobian(target)

            gradient = J.T @ W @ F

            objective = F.T @ W @ F
            objective_function.append(objective)

            if objective < 0.0001:# or objective > 2*objective_function[0]:
                msg = "Stopping because convergence" if objective < 0.0001 else "explosion of loss"
                print(msg + f": {objective}")
                plotter.update({'loss': objective, 'param': self.param, 'phase_space': F})
                break

            if search_algorithm == 'momentum':
                v = v * alpha + (1 - alpha) * gradient
                self.param -= learning_rate * v

            elif search_algorithm == 'adam':
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient) ** 2

                m_ = m / (1 - beta1 ** t)
                v_ = v / (1 - beta2 ** t)

                self.param -= learning_rate * m_ / (np.sqrt(v_) + epsilon)
            else:
                self.param -= learning_rate * gradient

            if plotter:
                plotter.update({'loss': objective, 'param': self.param, 'phase_space': F})

            params = np.vstack((params, self.param))

        return params, objective_function

    def sim(self, param):
        i_func = lambda t: self.input_func(t, [self.t0, self.t0 + self.T], param,
                                           start_end=(0.0, self.func_parameters.get('target_position', 0.0)))
        res = sci.odeint(func=self.system.create_simfunction(input_function=i_func, use_sp2c=True),
                         y0=self.x0, t=self.t)

        return res
    
    def generate(self, n, division):
        """
        Parameters
        ----------
        n : int
            Number of sample to be generated.

        division : string, 'random' or 'grid', default='random'
            Algorithm used to generate the samples based on the free parameters

        Returns
        -------
        x : array, shape = (n, state_dimension)
        y :
        res :

        """
        self.x = np.zeros((n, len(self.x0)))
        self.y = np.zeros((n, self.func_parameters['n']))
        pbar = tqdm.tqdm_notebook(total=n)            

        if division == 'random':
            i = 0
            while i < n:
                param = np.hstack((np.random.uniform(*span) for span in self.func_parameters['range']))
                res = self.sim(param)
                x_final = res[-1, :]

                if not self.inside_search_space(x_final):
                    continue

                self.x[i, :] = x_final
                self.y[i, :] = param
                i += 1
                pbar.update(1)

        elif division == 'grid':
            pass

        pbar.close()

        return self.x, self.y
    
    def inside_search_space(self, x_final):
        if self.search_space:
            return all(self.search_space[i][0] < x < self.search_space[i][1] for i, x in enumerate(x_final))
        else:
            return True


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
                continue

            new_data = dict()
            if self.i == 0:

                if fig.title.text == 'phase_space':
                    fig.xaxis.axis_label = 'phi'
                    fig.yaxis.axis_label = 'phi_dot'
                    new_data['x'], new_data['y'], *_ = [[data[0]], [data[2]]]

                else:
                    fig.xaxis.axis_label = 'iteration'
                    new_data['x'] = [self.i]
                    new_data['y'] = [data]
            else:
                if fig.title.text == 'phase_space':
                    new_data['x'] = plot.data_source.data['x'] + [data[0]]
                    new_data['y'] = plot.data_source.data['y'] + [data[2]]
                else:
                    new_data['x'] = plot.data_source.data['x'] + [self.i]
                    new_data['y'] = plot.data_source.data['y'] + [data]

            plot.data_source.data.update(new_data)
        self.i += 1

        push_notebook(handle=self.plot_handle)

    def draw(self):
        pass