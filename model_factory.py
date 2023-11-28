"""Python file to instantite the model and the transform that goes with it."""
from model import Net, ResNet50, ViTh14, ViTl16, ViTb16, EfficientNet
from data import data_transforms, data_transforms_r50, data_transforms_vith14, data_transforms_vitl16, data_transforms_vitb16


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet_pretrained":
            return ResNet50()
        elif self.model_name == "vit_h_14":
            return ViTh14()
        elif self.model_name == "vit_l_16":
            return ViTl16()
        elif self.model_name == "vit_b_16":
            return ViTb16()
        elif self.model_name == "efficient_net":
            return EfficientNet
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "resnet_pretrained":
            return data_transforms_r50
        elif self.model_name == "vit_h_14":
            return data_transforms_vith14
        elif self.model_name == "vit_l_16":
            return data_transforms_vitl16
        elif self.model_name == "vit_b_16":
            return data_transforms_vitb16
        elif self.model_name == "efficient_name":
            return data_transforms_r50
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
