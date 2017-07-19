# Py-denseflow


This is a python port of denseflow, which extract the videos' frames and **optical flow images** with **TVL1 algorithm** as default.

---

### Requirements:
- numpy
- cv2
- PIL.Image
- multiprocess
- scikit-video (optional)
- scipy

## Installation
#### Install the requirements:
```
pip install -r requirements.txt

```

---

## Usage
The denseflow.py contains two modes including '**run**' and '**debug**'.


here 'debug' is built for debugging the video paths and video-read methods. ([IPython.embed](http://ipython.org/ipython-doc/dev/interactive/reference.html#embedding) suggested)

Just simply run the following code:

```
python denseflow.py --new_dir=denseflow_py --num_workers=1 --step=1 --bound=20 --mode=debug

```
While in 'run' mode, here we provide multi-process as well as multi-server with manually s_/e_ IDs setting.

for example:  server 0 need to process 3000 videos with 4 processes parallelly working:

```
python denseflow.py --new_dir=denseflow_py --num_workers=4 --step=1 --bound=20 --mode=run --s_=0 --e_=3000
```

---

Just feel free to let me know if any bugs exist.

