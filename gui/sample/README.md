# XIMEA-Python-Auto-Video-Recorder

I find it hard to find an easy-to-use yet powerful python script to record XIMEA Hyperspectral video. So I made one. The official doc of XIMEA Python is over-simple but this project will help you to concentrate on your work. Record and save, with right parameters. That's all.

## Here's some parameters worth mentioning:

- `--name`           Session Name. The only identifier of this task.
- `--root`           The root path of the output video.
- `--img_mode`       The image data format.
- `--fps FPS`        The fps of the output video.
- `--exposure`       Camera exposure.

By default, the camera will be set to auto exposure by me, you can also easily change it to a static value by uncomment the line related to exposure in the code.

You need to pass a new name to the program in the command line every time you start a new experiment, it will be the one and the only identifier to your task, and it will make a new directory using this name below the root dir.
