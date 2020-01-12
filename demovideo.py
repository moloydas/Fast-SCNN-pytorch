import os
import argparse
import torch
import time
import sys

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image as Imag
from utils.visualize import get_color_pallete

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy
from geometry_msgs.msg import Twist
import ros
from multiprocessing import Pool
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()

class ImageCVBridge(object):
    def __init__(self):
        rospy.loginfo("Changing all ROS topics to cv image")
        #self.image_sub=rospy.Subscriber("/image_publisher_1559912716887579484/image_raw",Image,self.imgcallback)
        #self.image_sub=rospy.Subscriber("/cv_camera/image_raw",Image,self.imgcallback)
        self.image_sub=rospy.Subscriber("/zed/left/image_raw_color",Image,self.imgcallback)

        self.bridge=CvBridge()
        self.image_data=None
        self.cv_image=None
        rospy.loginfo("All objects for ros to cv conversion initialised")
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(self.device)
        print('Finished loading model!')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
        self.mtx=np.array([[709.103066,0.000000,621.543559],[0.000000 ,709.978057 ,333.677376],[0.000000 ,0.000000, 1.000000]])

        self.dist=np.array([-0.163186 ,0.026619 ,0.000410 ,0.000569 ,0.000000])

    def imgcallback(self,data):
        self.image_data=data

    def demo(self):

        # image transform
        img=self.bridge.imgmsg_to_cv2(self.image_data,"bgr8")

        img=cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
        
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(2048,1024))

        image=Imag.fromarray(img)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
        
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, args.dataset)
        mask_cv = np.array(mask.convert('RGB'))

        cv2.imshow('Original',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.imshow('Maskop',cv2.cvtColor(mask_cv,cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        end_time=time.time()

    def do_work(self):
        r=rospy.Rate(20)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Maskop", cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            start_time=time.time()
            self.demo()
            r.sleep()
            end_time=time.time()
            print("Run time:",(end_time-start_time))

if __name__ == '__main__':
    rospy.init_node("image_extractor")
    obj=ImageCVBridge()
    obj.do_work()
