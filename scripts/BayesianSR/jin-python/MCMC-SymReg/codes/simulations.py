#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from funcs import Operator, Node
from funcs import grow, genList, shrink, upgOd, allcal, display, getHeight, getNum, numLT, upDepth, Express, fStruc
from funcs import ylogLike, newProp, Prop, auxProp
from bsr import BSR

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.stats import norm
import copy
import random
import time

# =============================================================================
# # y = 2.5x1^4 - 1.3x1^3 + 0.5x2^2 - 1.7x2
# =============================================================================

random.seed(1)
n = 30
x1 = np.random.uniform(-3, 3, n)
x2 = np.random.uniform(-3, 3, n)
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
train_data = pd.concat([x1, x2], axis=1)
train_y = 2.5 * np.power(train_data.iloc[:, 0],4) - 1.3 * np.power(train_data.iloc[:, 0],3)+0.5*np.power(train_data.iloc[:, 1],2)-1.7* train_data.iloc[:, 0]

K = 2
MM = 1000
my_bsr = BSR(K,MM)
my_bsr.fit(train_data, train_y)
