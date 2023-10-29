from main import *
from utils import load_session
from preprocess import resize_and_pad
from PIL import Image


class Inference:
    def __init__(self):
        self.image_size = IMAGE_SIZE
        self.conf_thres = 0.7
        self.iou_thres = 0.7
        self.session = load_session(PATH_MODEL)
        self.session.get_providers()

    def Detect(self, img):
        image, ratio, (padd_left, padd_top) = resize_and_pad(img, new_shape=self.image_size)

        img_norm = normalization_input(image)

        pred = infer(self.session, img_norm)
        pred = postprocess(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]

        paddings = np.array([padd_left, padd_top, padd_left, padd_top])
        pred[:, :4] = (pred[:, :4] - paddings) / ratio

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        return visualize(image, pred)
