# zerdax2
Chess position recognizer.

## Disclaimer
This program is a work in progress and may not work as expected.

## Installation
### Clone and install dependencies
```
$ git clone https://github.com/lucas-mior/zerdax2
$ cd zerdax2
$ pip install -r requirements.txt
```
### Compile filter library
```
$ make
```

## Usage
```
$ python ./zerdax2.py <image> [<image2> ...] [-v=LEVEL]
```
Resulting FEN is written to standard output aswell
as position diagram using characters.

## Input image example with detection of pieces and colors
![Input image example](https://github.com/lucas-mior/zerdax2/blob/master/example.png?raw=true)

## Utilities
- `algorithm.py`: Run algorithm on images passed as arguments without reading
                standard input for filenames and with default logging level.
- `draw.py`: Superimpose 2 images
- `fen.py`: Compress FEN given as argument and draw position using characters
- `objects.py`: Run piece detection and save results on image.
