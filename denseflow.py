import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed
import skvideo.io
import scipy.misc

#dataset='activity_net'
#data_root='/n/zqj/video_classification/data/'

def ToImg(raw_flow,bound):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(256/float(2*bound))
    return flow

def save_flows(flows,image,save_dir,num,bound):
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(os.path.join(data_root,new_dir,save_dir)):
        os.makedirs(os.path.join(data_root,new_dir,save_dir))
    #save the image
    save_img=os.path.join(data_root,new_dir,save_dir,'img_{:05d}.jpg'.format(num))
    scipy.misc.imsave(save_img,image)
    #save the flows
    save_x=os.path.join(data_root,new_dir,save_dir,'flow_x_{:05d}.jpg'.format(num))
    save_y=os.path.join(data_root,new_dir,save_dir,'flow_y_{:05d}.jpg'.format(num))
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    #embed()
    #flow_x_img.save(save_x)
    #flow_y_img.save(save_y)
    scipy.misc.imsave(save_x,flow_x_img)
    scipy.misc.imsave(save_y,flow_y_img)
    return 0

def dense_flow(augs):
    #video_path=os.path.join(videos_root,video_name) # for activity-net
    video_name,save_dir,step,bound=augs
    video_path=os.path.join(videos_root,video_name.split('_')[1],video_name)
    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    try:
        videocapture=skvideo.io.vread(video_path)
    except:
        print '{} read error! '.format(video_name)
        return 0
    print video_name
    if videocapture.sum()==0:
        print 'Could not initialize capturing',video_name
        exit()
    len_frame=len(videocapture)
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=0
    while True:
        #frame=videocapture.read()
        if num0>=len_frame:
            break
        frame=videocapture[num0]
        num0+=1
        if frame_num==0:
            image=np.zeros_like(frame)
            gray=np.zeros_like(frame)
            prev_gray=np.zeros_like(frame)
            prev_image=frame
            prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_BGR2GRAY)
            frame_num+=1

            step_t=step
            while step_t>1:
                #frame=videocapture.read()
                num0+=1
                step_t-=1
            continue

        image=frame
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        frame_0=prev_gray
        frame_1=gray

        ##default choose the tvl1 algorithm
        dtvl1=cv2.createOptFlow_DualTVL1()
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        save_flows(flowDTVL1,image,save_dir,frame_num,bound) #this is to save flows and img.
        #embed()
        prev_gray=gray
        prev_image=image
        frame_num+=1

        step_t=step
        while step_t>1:
            #frame=videocapture.read()
            num0+=1
            step_t-=1

# def get_video_list():   # if the dataset is Activity-net
#     videos=os.listdir(videos_root)
#     videos.sort()
#     return videos,len(videos)

def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in os.listdir(cls_path):
            video_list.append(video_)
    video_list.sort()
    return video_list,len(video_list)

# def refind(video_list):
#     find_root='/S2/MI/zqj/video_classification/data/ucf101/flows'
#     raw_list=video_list
#     for video_ in raw_list:
#         video_path=os.path.join(find_root,video_.split('.')[0])
#         if os.path.exists(video_path):
#             raw_list.remove(video_)
#     #embed()
#     return raw_list,len(raw_list)

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the optical flows")
    parser.add_argument('--dataset',default='ucf101',type=str)
    parser.add_argument('--data_root',default='/n/zqj/video_classification/data',type=str)
    parser.add_argument('--new_dir',default='flows',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--s_',default=0,type=int)
    parser.add_argument('--e_',default=13320,type=int)
    args = parser.parse_args()
    return args



if __name__ =='__main__':

    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)
    args=parse_args()
    data_root=os.path.join(args.data_root,args.dataset)
    videos_root=os.path.join(data_root,'videos')



    #specify the augments
    num_workers=args.num_workers
    step=args.step
    bound=args.bound
    s_=args.s_
    e_=args.e_
    new_dir=args.new_dir
    #get video list
    video_list,len_videos=get_video_list()
    video_list=video_list[s_:e_]
    #video_list,len_videos=refind(video_list)
    #embed()
    len_videos=e_-s_
    print 'find {} videos.'.format(len_videos)
    flows_dirs=[video.split('.')[0] for video in video_list]
    print 'get videos list done! '
    pool=Pool(num_workers)
    #embed()
    pool.map(dense_flow,zip(video_list,flows_dirs,[step]*len(video_list),[bound]*len(video_list)))
    #dense_flow((video_list[0],flows_dirs[0],step,bound))