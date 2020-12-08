# from guiparse import *
import os, sys
sys.path.append('..')
from gui.guiparse import argStation

parser = argStation('Multispectral Video Recorder')
###########################################################
################# IMPORTANT PARAMETERS ####################
# -- File and Session Setting -- #
parser.add_argument(
    "-u",
    "--upload", 
    action="store_true",
    help="Upload the file before exit.")

parser.add_argument(
    "-d",
    "--delete_after_upload", 
    action="store_true", 
    help="Delete the file after upload.")

parser.add_argument(
    '--name',
    type=str,
    default="Test",
    help='Session Name. The only identifier of this task.')

# -- Camera Setting -- #
parser.add_argument(
    '--img_mode',
    type=str,
    default="XI_RAW8",#"XI_RAW8",XI_RGB24
    help='The image data format.')

parser.add_argument(
    '--fps',
    type=float,
    default='30.0',
    help='The fps of the output video.')

parser.add_argument(
    '--exposure',
    type=int,
    default='10000',
    help='Camera exposure.')

if os.name == "posix":
    parser.add_argument(
        '--root',
        type=str,
        default=r"/mnt/e/Children_Multispectral",
        is_dir=True,
        help='The root path of the output video.')
elif os.name == "nt":
    parser.add_argument(
        '--root',
        type=str,
        default=r"E:\Children_Multispectral",
        is_dir=True,
        help='The root path of the output video.')    
else:
    parser.add_argument(
        '--root',
        type=str,
        default="/Volumes/Takanashi/Datasets/Children",
        is_dir=True,
        help='The root path of the output video.')
    

# root = tk.Tk()
# #creation of an instance
# app = Window(root, parser, main_func="record_mulispectral_video.py",one_off=False)
# root.geometry(app.geometry)
# #mainloop 
# root.mainloop() 
