import yolov5.detect as yolo


def find_pieces(img):
    pieces = yolo.run(weights="yolov5/runs/train/exp1/weights/best.pt",
                      source=img.BGR_name,
                      data="yolov5/zerdax2.yaml",
                      conf_thres=0.7,  # confidence threshold
                      iou_thres=0.45,  # NMS IOU threshold
                      max_det=32,  # maximum detections per image
                      save_txt=False,  # save results to *.txt
                      save_conf=True,  # save confidences in --save-txt labels
                      project='.',  # save results to project/name
                      name='exp',  # save results to project/name
                      )
    print(pieces)
    img.pieces = pieces.tolist()
    return img


def determine_colors(img):
    return img


if __name__ == "__main__":
    find_pieces()
