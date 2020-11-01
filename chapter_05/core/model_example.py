import torch
from models import PNet,RNet,ONet
import numpy as np

if __name__ == "__main__":

    #------------------------------------------------------------------------------------ PNet
    print('\n ----------------------------------------------------------')
    image_hight = 480
    image_width = 640
    stride = 2
    # grid_num = ((image_hight-12)*(image_width-12))/stride + 1
    grid_hight = int(np.floor((image_hight-12)/2)+1)
    grid_width = int(np.floor((image_width-12)/2)+1)
    grid_num = grid_hight * grid_width

    grid_num = int(grid_num)
    print('\n feature map grid num :{}, height :{}, width :{}  '.format(grid_num,grid_hight,grid_width))


    input = torch.randn([32, 3, image_hight,image_width])

    m_PNet = PNet()

    # print('\n',m_PNet)

    with torch.no_grad():
        label, offset = m_PNet(input)
    print('\n <PNet> output :')
    print(' <PNet> label  :',label.size())
    print(' <PNet> offset :',offset.size())
    print(' <PNet> feature map grid num :{}, height :{}, width :{}'.format(offset.size()[2]*offset.size()[3],offset.size()[2],offset.size()[3]))


    #------------------------------------------------------------------------------------ RNet
    print('\n ----------------------------------------------------------')
    input = torch.randn([56, 3, 24,24])

    m_RNet = RNet()

    # print('\n',m_RNet)

    with torch.no_grad():
        label, offset = m_RNet(input)
    print('\n <RNet> output :')
    print(' <RNet> label  :',label.size())
    print(' <RNet> offset :',offset.size())

    #------------------------------------------------------------------------------------ RNet
    print('\n ----------------------------------------------------------')
    input = torch.randn([23, 3, 48,48])

    m_ONet = ONet()

    # print('\n',m_ONet)

    with torch.no_grad():
        label, offset = m_ONet(input)
    print('\n <ONet> output :')
    print(' <ONet> label  :',label.size())
    print(' <ONet> offset :',offset.size())
