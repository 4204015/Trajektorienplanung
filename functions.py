#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import string
from inspect import Parameter, signature

letters = dict(enumerate(string.ascii_lowercase, 1))

def paraboloid(N=2):
    def f(*args):  
        return np.sum(arg**2 for arg in args)
    
    params = [Parameter(letters[n+1], Parameter.POSITIONAL_ONLY) for n in range(N)] 
    sig = signature(f)
    f.__signature__ = sig.replace(parameters=params)
    
    return f