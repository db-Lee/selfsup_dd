from algorithms import distill

from algorithms import pretrain_dc
from algorithms import pretrain_frepo
from algorithms import pretrain_krr_st

from algorithms import scratch
from algorithms import linear_eval
from algorithms import finetune
from algorithms import zeroshot_kd

def get_algorithm(name):
    return globals()[name]