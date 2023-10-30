from main import *
from utils import load_session
from preprocess import resize_and_pad
from PIL import Image
import cv2

class Inference:
    """
    An Inference class for detecting objects in images using a trained model.

    Attributes:
        image_size (int): The desired input image size for inference.
        conf_thres (float): Confidence threshold for object detection.
        iou_thres (float): Intersection over Union (IoU) threshold for object detection.
        session (object): The loaded inference session with a pre-trained model.

    Methods:
        Detect(img):
        Perform object detection on the input image and return the image with bounding boxes.

    """
    def __init__(self):
        """
        Initialize the Inference class with default parameters and a loaded model session.
        """
        self.image_size = IMAGE_SIZE
        self.conf_thres = 0.7
        self.iou_thres = 0.7
        self.session = load_session(PATH_MODEL)
        self.session.get_providers()

    def Detect(self, img):
        """
        Perform object detection on the input image.

        Args:
            img (ndarray): An input image in numpy array format.

        Returns:
            Image: A PIL Image with bounding boxes drawn around detected objects.
        """
        image, ratio, (padd_left, padd_top) = resize_and_pad(img, new_shape=self.image_size)

        img_norm = normalization_input(image)

        pred = infer(self.session, img_norm)
        pred = postprocess(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]

        paddings = np.array([padd_left, padd_top, padd_left, padd_top])
        pred[:, :4] = (pred[:, :4] - paddings) / ratio

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        return visualize(image, pred)
