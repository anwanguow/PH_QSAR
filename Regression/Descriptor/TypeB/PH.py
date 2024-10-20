from ripser import Rips
rips = Rips(maxdim=2)
from sklearn.base import TransformerMixin
import numpy as np
import collections
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import matplotlib.pyplot as plt

def dist_mat(t):
    x=np.loadtxt(t,dtype=float,usecols=(1), skiprows=2)
    y=np.loadtxt(t,dtype=float,usecols=(2),skiprows=2)
    z=np.loadtxt(t,dtype=float,usecols=(3),skiprows=2)
    Distance=np.zeros(shape=(len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            Distance[i][j]=np.sqrt(  ((x[i]-x[j])**2)  + ((y[i]-y[j])**2)  + ((z[i]-z[j]) **2)  )
    return Distance

__all__ = ["PI"]

class PI(TransformerMixin):

    def __init__(
        self,
        pixels=(20, 20),
        spread=None,
        specs=None,
        kernel_type="gaussian",
        weighting_type="linear",
        verbose=True,
    ):

        self.specs = specs
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        self.nx, self.ny = pixels

        if verbose:
            print(
                'PI(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                    pixels, spread, specs, kernel_type, weighting_type
                )
            )

    def transform(self, diagrams):
        if len(diagrams) == 0:
            return np.zeros((self.nx, self.ny))
        try:
            singular = not isinstance(diagrams[0][0], collections.Iterable)
        except IndexError:
            singular = False
        if singular:
            diagrams = [diagrams]
        dgs = [np.copy(diagram) for diagram in diagrams]
        landscapes = [PI.to_landscape(dg) for dg in dgs]

        if not self.specs:
            self.specs = {
                "maxBD": np.max([np.max(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
                "minBD": np.min([np.min(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
            }
        imgs = [self._transform(dgm) for dgm in landscapes]

        if singular:
            imgs = imgs[0]

        return imgs

    def _transform(self, landscape):
        maxBD = self.specs["maxBD"]
        minBD = min(self.specs["minBD"], 0)
        dx = maxBD / (self.ny)
        xs_lower = np.linspace(minBD, maxBD, self.nx)
        xs_upper = np.linspace(minBD, maxBD, self.nx) + dx
        ys_lower = np.linspace(0, maxBD, self.ny)
        ys_upper = np.linspace(0, maxBD, self.ny) + dx
        weighting = self.weighting(landscape)
        img = np.zeros((self.nx, self.ny))
        
        if np.size(landscape,1) == 2:
            
            spread = self.spread if self.spread else dx
            for point in landscape:
                x_smooth = norm.cdf(xs_upper, point[0], spread) - norm.cdf(
                    xs_lower, point[0], spread
                )
                y_smooth = norm.cdf(ys_upper, point[1], spread) - norm.cdf(
                    ys_lower, point[1], spread
                )
                img += np.outer(x_smooth, y_smooth) * weighting(point)
            img = img.T[::-1]
            return img
        else:
            spread = self.spread if self.spread else dx
            for point in landscape:
                x_smooth = norm.cdf(xs_upper, point[0], point[2]*spread) - norm.cdf(
                    xs_lower, point[0], point[2]*spread
                )
                y_smooth = norm.cdf(ys_upper, point[1], point[2]*spread) - norm.cdf(
                    ys_lower, point[1], point[2]*spread
                )
                img += np.outer(x_smooth, y_smooth) * weighting(point)
            img = img.T[::-1]
            return img

    def weighting(self, landscape=None):
        if landscape is not None:
            if len(landscape) > 0:
                maxy = np.max(landscape[:, 1])
            else: 
                maxy = 1

        def linear(interval):
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d

        def pw_linear(interval):

            t = interval[1]
            b = maxy / self.ny

            if t <= 0:
                return 0
            if 0 < t < b:
                return t / b
            if b <= t:
                return 1

        return linear

    def kernel(self, spread=1):
        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)
        return gaussian

    @staticmethod
    def to_landscape(diagram):
        diagram[:, 1] -= diagram[:, 0]
        return diagram

    def show(self, imgs, ax=None):
        ax = ax or plt.gca()
        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")
        # plt.savefig("pers_image.png", dpi=300, bbox_inches='tight')
        plt.show()

def PI_vector_h0(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    Totalmatrix=h0matrix
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h1(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h1matrix=diagrams[1]
    Totalmatrix=h1matrix
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h2(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h2matrix=diagrams[2]
    Totalmatrix=h2matrix
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h0h1(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h1matrix=diagrams[1]
    Totalmatrix=np.vstack((h0matrix,h1matrix))
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h0h2(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h0matrix,h2matrix))
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h1h2(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h1matrix=diagrams[1]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h1matrix,h2matrix))
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())

def PI_vector_h0h1h2(Filename, pixelx=100, pixely=100, myspread=2, myspecs={"maxBD": 2, "minBD":0}, showplot=False):
    D = dist_mat(Filename)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    h0matrix=diagrams[0][0:-1,:]
    h1matrix=diagrams[1]
    h2matrix=diagrams[2]
    Totalmatrix=np.vstack((h0matrix,h1matrix,h2matrix))
    pim = PI(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
    imgs = pim.transform(Totalmatrix)
   
    if showplot == True:
        pim.show(imgs)
        plt.show()
    return np.array(imgs.flatten())
