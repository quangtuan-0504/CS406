import cv2
from facenet_pytorch import MTCNN
import time

COLOR_BOX = (0, 0, 255)
COLOR_TEXT = (0, 0, 255)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIZE_TEXT = 1


class FaceDetectorMTCNN(object):

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        for box, prob, ld in zip(boxes, probs, landmarks):
            box = list(map(lambda x: int(x), box))

            # Draw rectangle on frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLOR_BOX, THICKNESS)

            # Write text
            cv2.putText(frame, str(prob), (box[2], box[1]), FONT, SIZE_TEXT, COLOR_TEXT, THICKNESS, cv2.LINE_AA)

        return frame

    def run(self, frame):
        """
            Run the FaceDetectorMTCNN and draw landmarks and boxes around detected faces
        """

        try:
            # detect face box, probability and landmarks
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

            # draw on frame
            self._draw(frame, boxes, probs, landmarks)
        except:
            print('THIS FRAME IS FAILURE!!!')

        return frame


if __name__ == "__main__":
    mtcnn = MTCNN()
    fcd = FaceDetectorMTCNN(mtcnn)
    fcd.run(None)