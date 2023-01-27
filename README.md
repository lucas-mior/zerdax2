# zerdax2
Chess position recognizer.

## Disclaimer
This program is a work in progress and may not work as expected.

## Installation
- Clone and install dependencies
```
$ git clone https://github.com/lucas-mior/zerdax2
$ cd zerdax2
$ pip install -r requirements.txt
```
- Download yolov5 from ultralytics, apply patch and install dependencies
```
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ patch < ../yolov5_zerdax2.diff
$ pip install -r requirements.txt
$ cd ..
```
- Compile filter library
```
$ make
```

## Usage
```
$ python ./zerdax2.py <image> [<image2> ...] [--loglevel=LEVEL]
```
Image filenames are also read from standard input.
Resulting FEN is written to standard output aswell
as position diagram using characters.
Type `<C-c>` to quit.

## Utilities
- `algorithm.py`: Run algorithm on images passed as arguments without reading
                standard input for filenames and with default logging level.
- `drawings.py`: Superimpose 2 images
- `fen.py`: Compress FEN given as argument and draw position using characters
- `lffilter.py`: Run a low pass filter on images.
- `yolo_wrap.py`: Run piece detection and save results on image.
