import face_detection_pb2
import face_detection_pb2_grpc

import cv2
import time
import argparse

import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

from concurrent import futures
import grpc


class FaceDetectionService(face_detection_pb2_grpc.FaceDetServiceServicer):

    def __init__(self, model_file: str, verbose: bool):
        self.model_file = model_file
        self.verbose = verbose

        self.sess, self.graph = load_tf_model('models/face_mask_detection.pb')
        # anchor configuration
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5

        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        self.anchors_exp = np.expand_dims(anchors, axis=0)

        self.id2class = {0: 'Mask', 1: 'NoMask'}

    def inference(self, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160)):
        '''
        Main function of detection inference
        :param image: 3D numpy array of image
        :param conf_thresh: the min threshold of classification probabity.
        :param iou_thresh: the IOU threshold of NMS
        :param target_shape: the model input size.
        :return:
        '''
        # image = np.copy(image)
        height, width, _ = image.shape
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0  # 归一化到0~1
        image_exp = np.expand_dims(image_np, axis=0)
        y_bboxes_output, y_cls_output = tf_inference(self.sess, self.graph, image_exp)

        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh=conf_thresh,
                                                     iou_thresh=iou_thresh,
                                                     )

        resp = face_detection_pb2.FaceDetResponse()
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            det_obj = face_detection_pb2.DetectedObj(lx=xmin, ly=ymin, rx=xmax, ry=ymax, score=conf)
            resp.detObjs.append(det_obj)

        return resp

    def predict(self, request: face_detection_pb2.FaceDetRequest, context) -> face_detection_pb2.FaceDetResponse:
        print("start to process request")
        raw_data = np.asarray(bytearray(request.imageData), dtype="uint8")
        image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        if image is None:
            return face_detection_pb2.FaceDetResponse()

        h, w, c = image.shape
        if h == 0 or w == 0:
            return face_detection_pb2.FaceDetResponse()

        return self.inference(image=image, conf_thresh=request.confThresh, target_shape=(260, 260))


def serve(port: int, model_file: str, verbose: bool) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
          ('grpc.max_send_message_length', 50 * 1024 * 1024),
          ('grpc.max_receive_message_length', 50 * 1024 * 1024)
      ])
    face_detection_pb2_grpc.add_FaceDetServiceServicer_to_server(
        FaceDetectionService(model_file, verbose), server)
    print('start to listen port:' + str(port))
    # ipv6 mode
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection Service")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=50051,
        help="Server listen port",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default='models/face_mask_detection.pb',
        help="tensorflow model for inference",
    )

    args = parser.parse_args()

    serve(args.port, args.model, args.verbose)



