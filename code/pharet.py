"""
This file is part of the PhaseRetrieval project.
Copyright 2015 David W. Hogg (SCDA) (NYU) (MPIA).
"""

from numpy.fft import rfftn, irfftn

class pharetModel:

    def __init__(self, data, imageshape):
        """
        Must initialize the data, and the shape of the reconstructed image.
        """
        self.datashape = None
        self.imageshape = imageshape
        self.set_data(data)

    def set_data(self, data):
        if self.datashape is None:
            self.datashape = data.shape
        assert self.datashape == data.shape
        self.data = data

    def set_real_image(self, image):
        assert self.imageshape == image.shape
        self.image = image
        self.ft = None

    def set_ft_image(self, ft):
        assert self.datashape == ft.shape
        self.ft = ft
        self.image = None

    def self.get_data(self):
        return self.data

    def get_real_image(self):
        if self.image is None:
            self.image = irfftn(self.ft, self.imageshape)
        return self.image

    def get_ft_image(self):
        if self.ft is None:
            self.ft = rfftn(self.image, self.imageshape)
        return self.ft

    def get_squared_norm_ft_image(self):
        ft = self.get_ft_image()
        return (ft * ft.conj()).real

    def get_score(self):
        return (self.get_squared_norm_ft_image() - self.get_data()) ** 2

    def __call__(self, image):
        self.set_image(image)
        return self.get_score()
