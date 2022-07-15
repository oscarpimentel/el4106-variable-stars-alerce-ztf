from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import pandas as pd
from dask import dataframe as dd

###################################################################################################################################################

def get_object(detections_ddf, labels_ddf, obj_name, band):
    df = detections_ddf.loc[obj_name].compute() # FAST
    df = df[df['fid']==band]
    days = df['mjd'].values
    sorted_indexs = np.argsort(days)
    days = days[sorted_indexs]
    obs = df['magpsf_corr'].values[sorted_indexs]
    obs_error = df['sigmapsf_corr'].values[sorted_indexs]
    c = labels_ddf.loc[obj_name].compute()['classALeRCE'].values[0] # FAST
    return days, obs, obs_error, c