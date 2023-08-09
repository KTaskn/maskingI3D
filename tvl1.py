import cv2
import argparse
from glob import glob
import os
import numpy as np


def flow2rgb(flow, frame):
    hsv = np.zeros((frame.shape[0], frame.shape[1], 3))
    hsv[...,1] = 255    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    try:
        # load directory path from command line
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir', type=str, required=True)
        parser.add_argument('--outdir', type=str, required=True)
        parser.add_argument('--ext', type=str, default='tif', required=False)
        args = parser.parse_args()
        directory_path = args.dir
        extension = args.ext
        outdir = args.outdir
        print('directory path: ', directory_path)
        print('extension: ', extension)
        print('output directory: ', outdir)
        
        paths_img = sorted(glob(os.path.join(directory_path, f'*.{extension}')))
        
        for path_prv, path_nxt in zip(paths_img[:-1], paths_img[1:]):
            name = os.path.basename(path_prv).split('.')[0]
            prv = cv2.imread(path_prv)
            nxt = cv2.imread(path_nxt)
            prv = cv2.cvtColor(prv, cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
            optical_flow = cv2.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(prv, nxt, None)
            flow[flow >= 20] = 20
            flow[flow <= -20] = -20
            # scale to [-1, 1]
            max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
            # flow is numpy ndarray
            flow = flow / max_val(flow)
            
            # write flow to file
            flow_out_path = os.path.join(outdir, "flows", f'{name}.npy')
            flowimg_out_path = os.path.join(outdir, "images", f'{name}.png')
            np.save(flow_out_path, flow)        
            cv2.imwrite(flowimg_out_path, flow2rgb(flow, prv).astype(np.uint8))
    except Exception as e:
        print(e)