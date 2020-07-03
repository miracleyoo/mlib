# mlib

This repository is a personal python library made by *miracleyoo*. It contains frequently used functions and classes in domains which I am and was working on.

I will try my best to make every functions and classes in this repo be fully tested under various cases, and make sure every module, every document, every class and function have its detailed and easy-to-understand comments and usage. However, since it is maintained personally, it is also inevitable to be imperfect and have some problems while using. Feel free to contact me using *issue* or directly send me email, I'll be very happy to hear from you.

Currently, it mainly contains the following areas:

## Sub-Modules

- **Basic utils**. like timer and logging setup.
- **File and folder**. Like make new directory while not destroy existing one, list directory with a bunch of conditions, backup file, stem and suffix direct output, and so on.
- **Language functional support**. Like property check for class input, lazy loader, some wrapper functions, etc.
- **Debug utils**. Like keep the context and dive into an iPython console when crash.
- **Deep Learning**. Deep learning related sub-modules, blocks, utils, etc.
- **CV**. Computer Vision things. Like face detection, image processing, video processing.
- **NLP**. Nature Language Processing modules.
- **Visualization**. Visualization of various data and summaries.
- **Threading codes.** 
- **Crawling related codes**.
- More on the way!

## Functions

1. As you can see in the library, all kinds of functions are divided into various corresponding folders, each of which is treated as submodules. You can import those functions using `from mlib.xxx.xxx import xxx`. 
2. The `__init__.py` function imports various modules which I use frequently, like `os`, `time`, `np`, `pd`, `tqdm`. It also contains some abbreviation of some sub-modules as well as some most commonly used functions. Most of these import are using **Lazy Import**, which means it will not really import those packages until you call them or tried to use `Tab` to do the auto-completion. Hence the loading is actually extremely fast, in my own desktop, the command `from mlib import *` costs 0.04 second.
3. I will try to guarantee that the library has version consistency and make sure your functions work in newer versions. But sometimes this might not be possible due to the structure-level reformat, but this will mainly happen in the first year. So if you are sensitive to this, maybe try it when it become better organized!  

It is still a little baby, but this is the repo I will constantly update and repair any bug. Feel free to try it and make PR! :-)