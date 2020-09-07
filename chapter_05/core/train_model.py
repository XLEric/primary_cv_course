#-*-coding:utf-8-*-
# date:2019-08-30
# Author: X.L.Eric
# function: model for train

import sys
sys.path.append('./')
from core.models import PNet,RNet,ONet,LossFn
from core.detect import *

def init_model():
    #------------------------------------------  init model for R-Net & O-Net
    p_model_path = './ckpt/P-Net_latest.pth'
    r_model_path = './ckpt/R-Net_latest.pth'
    o_model_path = './ckpt/O-Net_latest.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    m_PNet,m_RNet,m_ONet = create_XNet(device,p_model_path,r_model_path,o_model_path)

    pmm_det  = MtcnnDetector(pnet=m_PNet,rnet=None,onet=None,min_face_size=50,threshold=[0.5, 0.5, 0.6])
    rmm_det  = MtcnnDetector(pnet=m_PNet,rnet=m_RNet,onet=None,min_face_size=50,threshold=[0.5, 0.55, 0.6])

    # print('pmm_det',pmm_det)
    return pmm_det,rmm_det

pmm_det,rmm_det = init_model()
