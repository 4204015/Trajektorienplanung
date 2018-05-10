#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits import mplot3d

class PlotterLLM:
    def __init__(self, options, sort=True):
        self.options = options
        n = len(options)
        self.fig, self.axes = plt.subplots(nrows=n, ncols=1)
        self.axes = np.array(self.axes).flatten()
        self.sort = sort
    
    # --- plot options 2D ---
    
    def result(self, ax):
        ax.set_title("Lernergebnis")
        ax.grid(True)
        ax.plot(self.u, self.y, color=self.standard_colors[0])
        ax.plot(self.u, self.y_pred, '--', color=self.standard_colors[2])
        
        half_err = (self.y_pred - self.y) 
        upper = (self.y_pred + half_err).ravel()
        lower = (self.y_pred - half_err).ravel()
        ax.fill_between(self.u.ravel(), upper, lower,
                        color=self.standard_colors[2], alpha=0.2)

    def error(self, ax):
        ax.set_title("Absoluter Fehler")
        ax.grid(True)
        ax.plot(self.u, self.y_pred - self.y, color=self.standard_colors[0])
        ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    
    def validity_function(self, ax):
        ax.set_title("Zugehörigkeitsfunktionen")
        for idx, column in enumerate(self.llm.A):
            ax.plot(self.u, column, color=self.colors[idx])
            if self.llm.M_ <= 20:
                ax.annotate(f"{self.idx[idx]}", xy=(self.llm.Xi[:, idx],
                               np.max(column)), xytext=(-4,-20),
                            textcoords='offset points')#, color=self.colors[idx])  
            
    def models(self, ax):
        ax.set_title("Zugehörigkeitsfunktionen")
        ax.grid(True)
        for idx, (u, y, c) in enumerate(self.llm.local_model_gen()):
            ax.plot(u, y, color=self.colors[idx])
            
            markerline, stemlines, _ = ax.stem(self.llm.Xi[:, idx], c, '--')
            plt.setp(markerline, color=self.colors[idx], alpha=0.5)
            plt.setp(stemlines, color=self.colors[idx], alpha=0.5)
            
            if self.llm.M_ <= 20:
                ax.annotate(f"{self.idx[idx]}", xy=(self.llm.Xi[:, idx], c),
                            xytext=(-4,5), textcoords='offset points')
    
    def report(self, ax):
        m ,s = divmod(self.llm.training_duration, 60)
        ax.set_title(f"Bericht\nTrainingszeit: {m}:{s:.2f} | "
                     + "$rmse_{min}$=" + f"{np.min(self.llm.global_loss):.6e}")
        ax.grid(True)
        ax.plot(range(1, self.llm.M_), self.llm.global_loss/np.max(self.llm.global_loss),
                color=self.standard_colors[0], label="rmse")
        ax.set_ylabel("Nomierter RMSE")
        ax.set_xlabel("Anzahl Teilmodelle")
        ax.semilogy(True)
        ax.set_xlim(left=1)
    
        ax_ = ax.twinx()
        d = np.array(self.llm.split_duration)*1000
        ax_.plot(range(1, self.llm.M_), d, color=self.standard_colors[1])
        ax_.set_ylabel('Dauer pro Split in $ms$', color=self.standard_colors[1])
        ax_.tick_params('y', colors=self.standard_colors[1])

    # --- plot options 3D ---
    
    def result3D(self, ax):
        ax.set_title("Lernergebnis")
        ax.grid(True)     
        c = ax.contour(*self.u, self.y, 10, colors='black')
        c.collections[-1].set_label("true")
        cs = ax.contour(*self.u, self.y_pred, 10, cmap='viridis')
        norm = Normalize(vmin=np.min(self.y_pred), vmax=np.max(self.y_pred))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax)
        ax.legend()
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    def error3D(self, ax):
        ax.set_title("Absoluter Fehler")
        ax.grid(True)
        c = ax.contourf(*self.u, self.y_pred - self.y, 50, cmap='coolwarm')
        self.fig.colorbar(c, ax=ax)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    
    def validity_function3D(self, ax):
        ax.set_title("Zugehörigkeitsfunktionen")
        for idx, column in enumerate(self.llm.A):
            #ax.plot(self.u, column, color=self.colors[idx])
            k = self.u[0].shape[0]
            cs = ax.contour(*self.u, np.reshape(column, (k, k)), 3, cmap='viridis')
        
        norm = Normalize(vmin=np.min(column), vmax=np.max(column))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax)
        ax.scatter(self.llm.Xi[0,:], self.llm.Xi[1,:],
                   marker='x', color=self.standard_colors[0], label=r"$\xi$")
        ax.legend(loc=4, bbox_to_anchor=(0., 1.0, 1., .102))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        
        
    def models3D(self, ax):
        ax.set_title("Teilmodelle")
        for idx, r in enumerate(self.llm.model_range):
            r = list(zip(*r))
            ax.add_patch(patches.Rectangle(xy=r[0], width=r[1][0]-r[0][0], height=r[1][1]-r[0][1], fill=False)) 
        ax.scatter(self.llm.Xi[0,:], self.llm.Xi[1,:],
                   marker='x', color=self.standard_colors[0], label=r"$\xi$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc=4, bbox_to_anchor=(0., 1.0, 1., .102))
    
    def report3D(self, ax):
        self.report(ax)         
        
    # ----------------
    
    def plot(self):
        self.fig.set_size_inches((9, len(self.options)*2.5))
        for idx, option in enumerate(self.options):
            self.axes[idx].cla()
            if not self.projection_3d:
                getattr(self, option)(self.axes[idx])
            else:
                getattr(self, option + "3D")(self.axes[idx])
        self.fig.tight_layout()
    
    def update(self, u, y, llm):
        self.u = u
        self.y = y
        self.llm = llm
        
        self.projection_3d = isinstance(u, list)
        if not self.projection_3d:
            self.y_pred = np.reshape(llm.predict(u), (len(u), 1))
        else:
            X = np.vstack((u[0].flatten(), u[1].flatten())).T
            k = u[0].shape[0]
            self.y_pred = np.reshape(llm.predict(X), (k, k))
            self.y = np.reshape(y, (k, k))
            
        # define colors for each model
        self.standard_colors = plt.get_cmap('plasma')(np.linspace(0, 0.9, 4))
        self.colors = plt.get_cmap('inferno')(np.linspace(0, 0.9, llm.M_))
        
        if self.sort and self.llm.N <= 1:
            sort_idx = np.argsort(llm.Xi, axis=1)[0]
            self.idx = sort_idx
            llm.Xi = llm.Xi[:, sort_idx]
            llm.A = llm.A[sort_idx, :]
            llm.Theta = llm.Theta[sort_idx, :]
            self.llm.model_range = np.array(self.llm.model_range)[sort_idx]
            
        self.plot()
        #plt.show()
        