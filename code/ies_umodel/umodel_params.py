'''
@file: umodel_params.py
@author: qxLiu
@time: 2019/12/11 9:48
'''


from enum import Enum
import os

class TOPK(Enum):
	top5 = 5;  # lef-side is name, right-side is value
	top10 = 10;

# DEFAULT_USER_MODEL_DIR = os.path.dirname(__file__)+'/res_umodels/'
DEFAULT_USER_MODEL_DIR = os.path.join(os.path.dirname(__file__),'res_umodels')

