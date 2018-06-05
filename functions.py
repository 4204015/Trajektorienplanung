#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as scip


def sidestepping_spline_func_gen(kind, **kwargs):

    def sidestepping_spline(p, t_range, param):
        if kind in ['linear', 'nearest']:
            spline_func = lambda x, y: scip.interp1d(x, y, kind, fill_value=0.0, bounds_error=False)
        elif kind == 'cubic':
            spline_func = lambda x, y: scip.CubicSpline(x, y, bc_type=kwargs.get('bc_type', "natural"))
        else:
            raise ValueError(f"Unknown selection for parameter 'kind': {kind}")

        if len(param) > 2:
            position_param = param[0:2]
            param = param[2:]
            assert len(param) == len(position_param)
            x = [t_range[0], position_param[0], position_param[1],
                 t_range[1] - position_param[1], t_range[1] - position_param[0], t_range[1]]

        else:
            x = np.linspace(*t_range, len(param) + 4)

        return spline_func(x=x, y=[0, -param[0], param[1], -param[1], param[0], 0])(p)

    return sidestepping_spline


def piecewise(t, t_range, n, param):
    space = np.linspace(t_range[0], t_range[1], n + 1)
    for idx, sec in enumerate(space):
        if idx == 0:
            continue

        if space[idx - 1] <= t <= sec:
            return param[idx - 1]
        elif t > space[-1]:
            return 0.0

vpiecewise = np.vectorize(piecewise, excluded=['t_range', 'n', 'param'])
func1 = lambda p, t_range, param: vpiecewise(p, t_range=t_range, n=4, param=[-param[0], param[1], -param[1], param[0]])