#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import string
import numpy as np
from inspect import Parameter, signature

letters = dict(enumerate(string.ascii_lowercase, 1))


def paraboloid(N=2):
    def f(*args):
        return np.sum(arg ** 2 for arg in args)

    params = [Parameter(letters[n + 1], Parameter.POSITIONAL_ONLY) for n in range(N)]
    sig = signature(f)
    f.__signature__ = sig.replace(parameters=params)

    return f


def pulse(width, value=1):
    ''' Returns a vector of (width+1) integer ones. '''
    return (width+1)*[value]


def stepconv(vector, width):
    ''' Computes the discrete convolution of vector with a "unit"
        pulse of given width.

        Formula: result[i] = Sum[j=0 to width] 1*vector[i-j]
        Where 0 <= i <= len(vector)+width-1, and the "1*" is the value
        of the implied unit pulse function: pulse[j] = 1 for 0<=j<=width.
    '''
    result = width*[0] + vector;
    for i in range(len(vector)):
        result[i] = sum(result[i:i+width+1])
    for i in range(len(vector), len(result)):
        result[i] = sum(result[i:])
    return result


class ConstrainedRandom(object):
    def __init__(self, ranges=None, target=None, seed=None):
        self._rand = random.Random(seed)
        if ranges != None: self.setrange(ranges)
        if target != None: self.settarget(target)

    def setrange(self, ranges):
        self._ranges = ranges
        self._nranges = len(self._ranges)
        self._nmin, self._nmax = zip(*self._ranges)
        self._minsum = sum(self._nmin)
        self._maxsum = sum(self._nmax)
        self._zmax = [y-x for x,y in self._ranges]
        self._rconv = self._nranges * [None]
        self._rconv[-1] = pulse(self._zmax[-1])
        for k in range(self._nranges-1, 0, -1):
            self._rconv[k-1] = stepconv(self._rconv[k], self._zmax[k-1])

    def settarget(self, target):
        self._target = target

    def next(self, target=None):
        k = target if target != None else self._target
        k = k - self._minsum;
        N = self._rconv[0][k]
        seq = self._rand.randint(0,N-1)
        result = self._nranges*[0]
        for i in range(len(result)-1):
            cv = self._rconv[i+1]
            r_i = 0
            while k >= len(cv):
                r_i += 1
                k -= 1
            while cv[k] <= seq:
                seq -= cv[k]
                r_i += 1
                k -= 1
            result[i] = r_i
        result[-1] = k # t
        return [x+y for x,y in zip(result, self._nmin)]
