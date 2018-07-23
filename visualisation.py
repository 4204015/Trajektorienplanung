import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from bokeh.io import push_notebook
from bokeh.models.glyphs import Rect, Line, Ellipse
from bokeh.models.renderers import GlyphRenderer
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Label
from bokeh.plotting import figure


class PendelWagenSystem:
    def __init__(self, car_width=0.1, car_height=0.05, rod_length=0.5, pendulum_size=0.05):
        self.car_width = car_width
        self.car_height = car_height

        self.rod_length = rod_length
        self.pendulum_size = pendulum_size

        self.pendulum = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0], y=[0])),
                                      glyph=Ellipse(x='x', y='y', width=self.pendulum_size, height=self.pendulum_size))

        self.car = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0], y=[0])),
                                 glyph=Rect(x='x', y='y', width=self.car_width, height=self.car_height))

        self.rod = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0,0], y=[0,0])),
                                 glyph=Line(x='x', y='y', line_width=2))

        self.move = GlyphRenderer(data_source=ColumnDataSource(dict(x=deque([0,1], maxlen=200), y=deque([0,1], maxlen=200))),
                                  glyph=Ellipse(x='x', y='y', width=0.008, height=0.008,
                                                fill_alpha=0.25, line_alpha=0.0, fill_color="#cc0000"))

        self.ground = GlyphRenderer(data_source=ColumnDataSource(dict(x=[-100, 100], y=[-car_height/2, -car_height/2])),
                                    glyph=Line(x='x', y='y', line_width=1, line_alpha=0.5))

    def draw(self, state):
        phi, q = state[[0, 1]]
        pendulum_x = -self.rod_length * np.sin(phi) + q
        pendulum_y = self.rod_length * np.cos(phi)
        self.pendulum.data_source.data['x'] = [pendulum_x]
        self.pendulum.data_source.data['y'] = [pendulum_y]
        self.car.data_source.data['x'] = [q]
        self.rod.data_source.data['x'] = [q, pendulum_x]
        self.rod.data_source.data['y'] = [0, pendulum_y]

        new_move_data = dict(x=self.move.data_source.data['x'] + deque([pendulum_x]),
                             y=self.move.data_source.data['y'] + deque([pendulum_y]))

        self.move.data_source.data.update(new_move_data)

    @property
    def glyphs(self):
        return [self.car, self.rod,self.pendulum, self.move, self.ground]


class Animation:
    """
    Provides animation capabilities.

    Given a callable function that draws an image of the system state and smiulation data
    this class provides a method to created an animated representation of the system.
    """

    def __init__(self, sim_data, image=PendelWagenSystem(), speed_factor=1, fig=None):

        self.sim_data = sim_data
        self.x, self.t = sim_data

        self.figure = fig if fig else self.create_fig()
        self.image = image

        k = int(self.x.shape[0] / (np.max(self.t)/0.01))

        self.x = self.x[0:-1:k]
        self.t = self.t[0:-1:k]

        self.speed_factor = speed_factor

        self.time = Label(x=25, y=self.figure.plot_height - 50, x_units='screen', y_units='screen')
        self.figure.add_layout(self.time)
        self.figure.renderers += self.image.glyphs

    def animate(self, target):

        speed_string = f'  ({self.speed_factor}x)' if not self.speed_factor == 1 else ""
        for i, t in enumerate(self.t):
            self.time.text = f't = {np.around(t, decimals=2):.3f}' + speed_string
            self.image.draw(self.x[i, :])

            #if self.fps_sim_data > self.fps:
            #    time.sleep(1/(self.fps_sim_data - self.fps))

            time.sleep(0.01 / self.speed_factor)

            push_notebook(handle=target)

    def create_fig(self):
        # adapt plot ranges

        h = 0
        w = 1000

        while h == 0 or h > 800:

            w -= 100

            x_range = [np.min(self.x[:, 1]) - 0.1, np.max(self.x[:, 1]) + 0.1]
            y_range = [-0.5 - 0.1, 0.5 + 0.1]

            h = int((np.diff(y_range)[0]/np.diff(x_range)[0])*w)

        return figure(x_range=x_range, y_range=y_range, width=w, height=h)

    def show(self, t):
        pass

    def set_limits(self):
        pass

    def set_labels(self):
        pass


def plot_sim_states(res, size=(12, 12)):
    X1, X2, X3, X4 = res[0].T
    tt = res[1]
    
    plt.figure(figsize=size)
    plt.subplot(2, 2, 1)
    plt.plot(tt, np.rad2deg(X1))
    plt.ylabel(r"$\varphi$ in °")
    plt.xlabel("t in s")
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(tt, X2)
    plt.ylabel("$q$ in m")
    plt.xlabel("t in s")
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(tt, X3)
    plt.ylabel(r"$\dot{\varphi}$ in $\frac{°}{s}$")
    plt.xlabel("t in s")
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(tt, X4)
    plt.ylabel(r"$\dot{q}$ in $\frac{m}{s}$")
    plt.xlabel("t in s")
    plt.grid()
    plt.tight_layout()