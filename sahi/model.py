# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import warnings
from typing import Dict, List, Optional, Union
import importlib
import torch
import numpy as np
from yolox.data.datasets import COCO_CLASSES

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.torch import cuda_is_available, empty_cuda_cache

from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

logger = logging.getLogger(__name__)


class DetectionModel:
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        nms_threshold = 0.4,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
        classes = COCO_CLASSES,

    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.classes = classes
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        # automatically set device if its None
        if not (self.device):
            self.device = "cuda:0" if cuda_is_available() else "cpu"

        # automatically load model if load_at_init is True
        if load_at_init:
            self.load_model()

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        empty_cuda_cache()

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """
        NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,image: np.ndarray = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
        image_size: int = None
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        assert self.category_remapping is not None, "self.category_remapping cannot be None"
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,image: np.ndarray = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
        image_size: int = None
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            image = image,
            image_size=image_size,
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions



class YoloXDetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolox
        except ImportError:
            raise ImportError("Please run pip install -U yolox")
        current_exp = importlib.import_module(self.config_path)
        exp = current_exp.Exp()
        
        model = exp.get_model()
        model.cuda()
        model.eval()
        #print(model)
        self.model = model
        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        if not self.category_mapping:
              category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
              self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        try:
            import yolox
        except ImportError:
            raise ImportError('Please run "pip install -U yolox" ' "to install YOLOX first for YOLOX inference.")

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        preproc = ValTransform(legacy = False)
        if image_size is not None:
            tensor_img, _ = preproc(image, None, image_size)
        elif self.image_size is not None:
            tensor_img, _ = preproc(image, None, self.image_size)
        else:
            tensor_img, _ = preproc(image, None, (256,256))
        
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
        tensor_img = tensor_img.float()
        tensor_img = tensor_img.cuda()

        with torch.no_grad():
            prediction_result = self.model(tensor_img)
            prediction_result = postprocess(
                    prediction_result, len(self.classes), self.confidence_threshold,
                    self.nms_threshold, class_agnostic=True
                )
        
        if (prediction_result[0] is not None):
            prediction_result = prediction_result[0].cpu()
            bboxes = prediction_result[:,0:4]
            if image_size is not None:
                bboxes /= min(image_size[0] / image.shape[0], image_size[1] / image.shape[1])
            elif self.image_size is not None:
                bboxes /= min(self.image_size[0] / image.shape[0], self.image_size[1] / image.shape[1])
            else:
                bboxes /= min(256 / image.shape[0], 256 / image.shape[1])

            prediction_result[:,0:4] = bboxes

        self._original_predictions = prediction_result

    @property
    def category_names(self):
        return self.classes

    def _create_object_prediction_list_from_original_predictions(
        self,
        image: np.ndarray,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
        image_size: int = None
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        bboxes=[]
        bbclasses=[]
        scores=[]

        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]
        
        if(original_predictions[0] is not None):
            bboxes = original_predictions[:,0:4]
            bbclasses = original_predictions[:, 6]
            scores = original_predictions[:, 4] * original_predictions[:, 5]        

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list_per_image = []
        object_prediction_list = []

        for ind in range(len(bboxes)):
              box = bboxes[ind]
              cls_id = int(bbclasses[ind])
              score = scores[ind]
              if score < self.confidence_threshold:
                continue
              
              x0 = int(box[0])
              y0 = int(box[1])
              x1 = int(box[2])
              y1 = int(box[3])

              bbox = [x0,y0,x1,y1]

              object_prediction = ObjectPrediction(
                bbox = bbox,
                category_id=cls_id,
                bool_mask=None,
                category_name=self.category_mapping[str(cls_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
              object_prediction_list.append(object_prediction)
        
        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image
