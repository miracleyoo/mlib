# Author: Zhongyang Zhang
# E-mail: mirakuruyoo@gmail.com
import pickle as pkl
import time
import datetime

import os
import cv2
import argparse
import subprocess
# from guiparse import *
from ximea import xiapi
from pathlib2 import Path
from mlib import gui

ONE_CHANNEL_FORMAT = ["XI_RAW16","XI_RAW8","XI_MONO16","XI_MONO8"]
THREE_CHANNEL_FORMAT = ["XI_RGB24"]

# Log function
def log(*snippets, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]),
              end=end)

# Build the data storage structure
def build_data_repo_structure(args):
    root = Path(args.root)
    if not root.exists():
        root.mkdir()
    data_path = root / args.name
    if not data_path.exists():
        data_path.mkdir()
    temp = Path(os.getcwd(), "temp")
    if not temp.exists():
        temp.mkdir()
    return data_path

# Start the multispectral camera recording and save it to file. 
def start_recording(args):
    # Create instance for first connected camera
    cam = xiapi.Camera()

    # Build the data storage sturcture and get the data path
    data_path = build_data_repo_structure(args)

    # Start communication
    log('Opening first camera...')
    # To open specific device, you need to find an 8 digit serial 
    # number on the camera, and use: 
    cam.open_device_by_SN('28780454')
    # (open by serial number)
    # cam.open_device()

    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
    cam.set_framerate(args.fps)
    log('Frame rate is %f.' % cam.get_framerate())

    # Set record image dataframe
    cam.set_imgdataformat(args.img_mode)
    log('Image data format is %s.' % cam.get_imgdataformat())

    # Enable auto exposure
    cam.enable_aeag()
    # Set certain exposure value
    # cam.set_exposure(args.exposure)
    log("Automatic exposure enabled!")
    log('Current exposure is %s us.' % cam.get_exposure())

    # Enable Automatic White Balance
    # cam.enable_auto_wb()
    # log("Automatic white balance enabled!")

    log('Current frame width and height is: Width:{}, Height:{}'.format(cam.get_width(), cam.get_height()))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    output_name = str(data_path / (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.avi'))
    log(output_name)
    if args.img_mode in ONE_CHANNEL_FORMAT:
        out = cv2.VideoWriter(output_name, 0, args.fps, (cam.get_width(), cam.get_height()), 0)
    elif args.img_mode in THREE_CHANNEL_FORMAT:
        out = cv2.VideoWriter(output_name, 0, args.fps, (cam.get_width(), cam.get_height()))

    # Create instance of Image to store image data and metadata
    img = xiapi.Image()

    # Start data acquisition
    log('Starting data acquisition...')
    cam.start_acquisition()

    try:
        log('Starting video. Press CTRL+C to exit.')
        t0 = time.time()
        while True:
            # Get data and pass them from camera to img
            cam.get_image(img)

            # Create numpy array with data from camera. Dimensions of the array are
            # determined by imgdataformat
            frame = img.get_image_data_numpy()
            # log(frame.shape)

            # Write the frame into the output file
            out.write(frame.astype('uint8'))

            # Show acquired image with time since the beginning of acquisition
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = '{:5.2f}'.format(time.time()-t0)

            winname = "Multispectral Camera Video Recording"
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
            cv2.moveWindow(winname, 0, 0)
            cv2.resizeWindow(winname, 1024, 544)

            cv2.putText(
                frame, text, (900, 150), font, 4, (255, 255, 255), 2
            )
            
            cv2.imshow(winname, frame) #cv2.resize(frame,(1024, 544)))
            
            cv2.waitKey(1)

    except KeyboardInterrupt:
        out.release()
        cv2.destroyAllWindows()

    # Stop data acquisition
    log('Stopping acquisition...')
    cam.stop_acquisition()

    # Stop communication
    cam.close_device()

    # Upload the file to remote server.
    if args.upload:
        if args.delete_after_upload:
            if os.name in ["posix","nt"]:
                os.system("wsl python3 ./wsl-jump.py -d -o "+output_name.replace("\\","\\\\"))
            else:
                os.system("nohup python3 ./upload_video.py -d -f "+output_name+" > ./temp/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_upload.log"+" 2>&1 &")
        else:
            if os.name in ["posix","nt"]:
                os.system("wsl python3 ./wsl-jump.py -o "+output_name.replace("\\","\\\\"))
            else:
                os.system("nohup python3 ./upload_video.py -f "+output_name+" > ./temp/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_upload.log"+" 2>&1 &")

if __name__ == "__main__":
    # with open("./gui/args.pkl","rb") as f:
    #     args=dotdict(pkl.load(f))
    args = gui.load_args()
    start_recording(args)
