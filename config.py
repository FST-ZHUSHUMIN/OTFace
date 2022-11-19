import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        DATA_ROOT = '/home/shumin/zhushumin-project/OTFace/dataset/ms1m_align_112/imgs/' , # the parent root where your train/val/test data are stored train= '/zhushumin/ms1m_align_112'  /test/dataface/datasets/Data_Zoo
        # DATA_ROOT = '/home/shumin/Documents/new_folder/zhushumin-project/OTFace/dataset/CASIA-WebFace-112/',
        # RECORD_DIR = '/data2/yugehuang/data/refined_ms1m.txt', # the dataset record dir
        MODEL_ROOT = './train_log/model/otface_arcface/', # the root to buffer your checkpoints
        LOG_ROOT = './train_log/log/otface_arcface/', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = '',
        HEAD_RESUME_ROOT = "",
        BACKBONE_NAME = 'Mobileface', # support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'OTFace_Arcface', # support:  ['AMsoftmax', 'ArcFace', 'MV_AMsoftmax_a', 'MV_ArcFace_a', 'OTFace_Arcface', 'OTFace_AMsoftmax']
        LOSS_NAME = 'Softmax', # support: ['Focal', 'Softmax']
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.498, 0.498, 0.498], # for normpalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        LR = 0.1, # initial LR
        START_EPOCH = 0, #start epoch
        NUM_EPOCH = 25, # total epoch number ms1mv2[24], casia_webface[50]  
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [10,18,22], # ms1mv2[10, 18, 22] #casia_webface[28, 38, 46] 
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = 0, # specify your GPU ids
        DIST_BACKEND = 'nccl',
        DIST_URL = 'tcp://localhost:13456',
        NUM_WORKERS = 8,
        TEST_GPU_ID = [0]
    ),
}
