############################## Import Dependencies ##############################
import os
from easydict import EasyDict
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
# 导入我们重构后的模型架构
from .arch import DMID

############################## Hot Parameters ##############################
# 数据集与评估指标配置
DATA_NAME = 'PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# 模型架构与参数
MODEL_ARCH = DMID   # 指定模型为我们新创建的DMID
MODEL_PARAM = {
    # 基础参数
    "num_nodes": 170,
    "input_len": INPUT_LEN,
    "input_dim": 3,
    "embed_dim": 64,
    "output_len": OUTPUT_LEN,
    "num_layer": 3,
    # 时间嵌入参数 (与STID对齐，使用可学习参数)
    "if_T_i_D": True,
    "if_D_i_W": True,
    "temp_dim_tid": 32,     # 16, 24, 32, 48, 64
    "temp_dim_diw": 32,     # 16, 24, 32, 48, 64
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    # 空间嵌入参数 (DMID核心创新与配置开关)
    "if_node": True,                           # 空间嵌入总开关
    "if_dmid_spatial": True,                   # **True: 开启DMID创新空间嵌入, False: 回退到STID原始嵌入**
    "use_manifold_similarity": True,           # **True: 在相似性嵌入上使用流形学习(球面投影), False: 使用普通欧氏空间嵌入**
    "identity_dim": 24,                        # 确定性身份嵌入(傅里叶特征)的维度, 16, 24, 32, 48
    "similarity_dim": 48,                      # 可学习相似性嵌入的维度（投影前）, 24, 32, 40, 48, 64, 72
    "spatial_combination_method": 'gated_add',    # 嵌入组合方式: 'concat' 或 'gated_add'
    # STID原始空间嵌入参数 (仅在 if_dmid_spatial=False 且 if_node=True 时生效)
    "node_dim_stid": 64,
    "dropout": 0.1,
}
NUM_EPOCHS = 150

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'Refactored DMID model with deterministic identity and manifold similarity on PEMS08'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.DETERMINISTIC = True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0003,
}

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [7, 18, 25, 50, 80, 125],
    "gamma": 0.75
}

# CFG.TRAIN.CLIP_GRAD_PARAM = {
#     'max_norm': 5.0
# }

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation and Test Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
