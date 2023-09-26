from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C
def add_food_config(cfg):
    _CC = cfg
    # ----------- Backbone ----------- #
    _CC.MODEL.BACKBONE.FREEZE = False
    _CC.MODEL.BACKBONE.FREEZE_AT = 3

    # ------------- RPN -------------- #
    _CC.MODEL.RPN.FREEZE = False
    _CC.MODEL.RPN.ENABLE_DECOUPLE = False
    _CC.MODEL.RPN.BACKWARD_SCALE = 1.0

    # ------------- ROI -------------- #
    _CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
    _CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
    _CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
    _CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
    _CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
    _CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
    _CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.5
    _CC.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 40
    _CC.MODEL.ROI_HEADS.NUM_BASE_CLASSES = 20

    # register RoI output layer
    _CC.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS = "FastRCNNOutputLayers"

    _CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster
    # scale for cosine classifier
    _CC.MODEL.ROI_HEADS.COSINE_SCALE = 20
    # thresh for visualization results.
    _CC.MODEL.ROI_HEADS.VIS_IOU_THRESH = 1.0

    # ------------- TEST ------------- #
    _CC.TEST.PCB_ENABLE = False
    _CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
    _CC.TEST.PCB_MODELPATH = ""
    _CC.TEST.PCB_ALPHA = 0.50
    _CC.TEST.PCB_UPPER = 1.0
    _CC.TEST.PCB_LOWER = 0.05
    _CC.TEST.SAVE_FIG = False
    _CC.TEST.SCORE_THREHOLD = 0.5
    _CC.TEST.IMG_OUTPUT_PATH = ""
    _CC.TEST.SAVE_FEATURE_MAP = False

    # ------------ Other ------------- #
    _CC.SOLVER.WEIGHT_DECAY = 5e-5
    _CC.MUTE_HEADER = True
    # _CC.SOLVER.OPTIMIZER = 'SGD'

    # unknown probability loss
    _CC.UPLOSS = CN()
    _CC.UPLOSS.ENABLE_UPLOSS = False
    _CC.UPLOSS.START_ITER = 0  # usually the same as warmup iter
    _CC.UPLOSS.SAMPLING_METRIC = "min_score"
    _CC.UPLOSS.TOPK = 3
    _CC.UPLOSS.SAMPLING_RATIO = 7
    _CC.UPLOSS.ALPHA = 1.0
    _CC.UPLOSS.WEIGHT = 0.5

    # evidential loss
    _CC.ELOSS = CN()
    _CC.ELOSS.ENABLE_ELOSS = False
    _CC.ELOSS.WEIGHT = 0.1

    # hsic loss
    _CC.HSICLOSS = CN()
    _CC.HSICLOSS.ENABLE_HSICLOSS = False

    # instance contrastive loss
    _CC.ICLOSS = CN()
    _CC.ICLOSS.OUT_DIM = 128
    _CC.ICLOSS.QUEUE_SIZE = 256
    _CC.ICLOSS.IN_QUEUE_SIZE = 16
    _CC.ICLOSS.BATCH_IOU_THRESH = 0.5
    _CC.ICLOSS.QUEUE_IOU_THRESH = 0.7
    _CC.ICLOSS.TEMPERATURE = 0.1
    _CC.ICLOSS.WEIGHT = 0.1

    # swin transformer
    _CC.MODEL.SWINT = CN()
    _CC.MODEL.SWINT.EMBED_DIM = 96
    _CC.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    _CC.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    _CC.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    _CC.MODEL.SWINT.WINDOW_SIZE = 7
    _CC.MODEL.SWINT.MLP_RATIO = 4
    _CC.MODEL.SWINT.DROP_PATH_RATE = 0.2
    _CC.MODEL.SWINT.APE = False
    _CC.MODEL.SWINT.IN_CHANNELS = 768
    _CC.MODEL.BACKBONE.FREEZE_AT = -1
    _CC.MODEL.FPN.TOP_LEVELS = 2

