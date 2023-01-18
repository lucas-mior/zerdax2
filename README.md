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
Type `<C-c>` to quit.
