# zerdax2
Chess position recognizer.

## Disclaimer
This program is a work in progress and may not work as expected.

## Installation
- Download yolov5 from ultralytics
```
$ git clone https://github.com/ultralytics/yolov5
```

- Clone and chdir to this repo
```
$ git clone https://github.com/lucas-mior/zerdax2
$ cd zerdax2
```

- Install dependencies
```
$ pip install -r requirements.txt
```

- Link yolov5
```
$ ln -s ../yolov5 ./
```

## Usage
```
$ ./zerdax2.py <image> [<image2> ...] [--loglevel=LEVEL]
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
