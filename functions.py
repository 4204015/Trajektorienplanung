#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import numpy as np
import sympy as sp
import scipy.interpolate as scip


def sidestepping_spline_func_gen(kind, position=False, start_end=(0, 0), diff=0, **kwargs):

    def sidestepping_spline(t, t_range, param, start_end=start_end):
        if kind in ['nearest']:
            spline_func = lambda x, y: scip.interp1d(x, y, kind, fill_value=0.0, bounds_error=False)
        elif isinstance(kind, int):
            spline_func = lambda x, y: scip.make_interp_spline(x, y, k=kind, bc_type=kwargs.get('bc_type'))
        else:
            raise ValueError(f"Unknown selection for parameter 'kind': {kind}")

        if position:
            warnings.warn('', category=DeprecationWarning)
            s = int(len(param) / 2)
            position_param = param[0:s]
            param = param[s:]
            assert len(param) == len(position_param)
            x = [t_range[0], position_param[0], position_param[1],
                 t_range[1] - position_param[1], t_range[1] - position_param[0], t_range[1]]

        else:
            x = np.linspace(*t_range, len(param) * 2 + 2)

        param = np.array(param)
        func = spline_func(x=x, y=[start_end[0], *param + np.mean(start_end),
                                   *-param[::-1] + np.mean(start_end), start_end[1]])
        if isinstance(kind, int):
            return func.derivative(nu=diff)(t)
        else:
            return func(t)

    return sidestepping_spline


def spline_func_gen(kind, start_end=(0.0, 0.0), diff=0, symbolic=False, **kwargs):

    def spline(t, t_range, param, start_end=start_end):

        x = np.linspace(*t_range, len(param) + 2)
        param = np.array(param)

        scipy_spline = scip.make_interp_spline(x=x, y=[start_end[0], *param, start_end[1]], k=kind, bc_type=kwargs.get('bc_type'))

        if not symbolic:

            if diff != 0:
                return scipy_spline.derivative(nu=diff)(t) if t <= t_range[1] else 0
            else:
                return scipy_spline(t) if t <= t_range[1] else 0

        else:
            t = sp.Symbol('t')
            knots, coeffs, degree = scipy_spline.t, scipy_spline.c, scipy_spline.k
            basis = sp.functions.bspline_basis_set(degree, knots, t)
            sympy_spline = sum([c * b for c, b in zip(coeffs, basis)])

            return sp.diff(sympy_spline, t, diff)

    return spline


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