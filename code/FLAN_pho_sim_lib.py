# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:35:59 2022

@author: Xu Yang
pip install opencv-contrib-python==3.4.2.16
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp, EuclideanTransform,SimilarityTransform, rotate,rescale
from skimage import transform,filters
# sigma=[2,-0.8,2,-0.8]for nir
# sigma=[4,-2,4,-2] for tio
def per_data(Sdo, High, maskH,maskS,sc0, lr, ud,K=0.9,sigma=[2,-2,2,-2]):
    from skimage import transform, filters
    import skimage.morphology as sm
    High = filters.gaussian(imnorm(High), sigma=sc0*0.5)*maskH

    def mask3sig(im,mask,sig=[0,0]):
        
        med=np.median(im[mask])
        t=im[mask].std()
        maskH2=(im<(med+sig[0]*t)) & (im>(med+sig[1]*t))
#        maskH2=(im<(med-2*t))
    
        mask=mask & maskH2
    
        return mask
    H=High[maskH]
    High=imnorm(High,mx=H.max(),mi=H.min())*maskH
    M = High.shape[0] // sc0
    N = High.shape[1] // sc0
    sHigh = transform.resize(High, (M, N),mode='reflect')
    
    smaskH= transform.resize(maskH*1.0, (M, N),mode='reflect')>0.9
    smaskH=sm.erosion(smaskH,sm.square(5)) 

    sHigh = filters.gaussian(sHigh, K)*smaskH
    
    maskS=mask3sig(Sdo,maskS,sigma[0:2])
    smaskH=mask3sig(sHigh,smaskH,sigma[2:4])

    H=sHigh[smaskH]
    sHigh = imnorm(sHigh,mx=H.max(),mi=H.min()) 

    S = imnorm(removenan(Sdo))*0.8
    
    S = filters.gaussian(S, 0.5)*maskS
    
    S = imnorm(S,mx=S[maskS].max(),mi=S[maskS].min())
    
    sHigh=sHigh*255
    S = S * 255
    sHigh = np.uint8(sHigh)
    S = np.uint8(S)
    
    sc0=0.5*(High.shape[0]*1.0/M+High.shape[1]*1.0/N)
    return S, sHigh,sc0

def per_gst(im0,maskim,sc0, lr, ud, sigma, K=0.8):
    from skimage import transform, filters
    import skimage.morphology as sm
    im0 = filters.gaussian(imnorm(im0), sigma=sc0*2)*maskim
    def mask3sig(im,mask,sig=[0,0]):
        
        med=np.median(im[mask])
        t=im[mask].std()
        maskH2=(im<(med+sig[0]*t)) & (im>(med+sig[1]*t))
#        maskH2=(im<(med-2*t))   
        mask=mask & maskH2
    
        return mask
    Im=im0[maskim]
    im0=imnorm(im0,mx=Im.max(),mi=Im.min())*maskim
    M = im0.shape[0] // sc0
    N = im0.shape[1] // sc0
    im1 = transform.resize(im0, (M, N),mode='reflect')
    
    maskim1= transform.resize(maskim*1.0, (M, N),mode='reflect')>0.9
    maskim1=sm.erosion(maskim1,sm.square(5)) 

    im1 = filters.gaussian(im1, K)*maskim1
    
    maskim1=mask3sig(im1,maskim1,sigma[2:4])
    
    Img=im1[maskim1]
    im1 = imnorm(im1,mx=Img.max(),mi=Img.min()) 
    im1=im1*255
    im1 = np.uint8(im1)

    sc0=0.5*(im0.shape[0]*1.0/M+im0.shape[1]*1.0/N)
    return im1,sc0

def siftImageAlignment(img1, img2, img3, Hsize, debug=0, mask1=None, mask2=None,KG=0.75,scale=None,san=2.0,img2Org=None):
    from skimage.feature import plot_matches
    from skimage.measure import ransac
    if scale is None:
        func=SimilarityTransform
    else:
        func=EuclideanTransform
    
    if img2Org is None: 
        img2Org=img2
    else:
        img2Org=imnorm(img2Org)
       
    def sift_kp(image, tt='Match_image', mask=None):
        if mask is not None:
            mask = np.uint8(mask * 255)
        sift = cv2.xfeatures2d_SIFT.create()
#        sift = cv2.xfeatures2d_SURF.create(200)
#        sift=cv2.ORB_create()
        kp, des = sift.detectAndCompute(image, mask)
        kp_image = cv2.drawKeypoints(image, kp, None)
        
        print(len(kp))
        plt.figure()
        plt.imshow(kp_image[::-1,:,:])
        plt.xlabel("X (pixel)",fontsize=15)
        plt.ylabel("Y (pixel)",fontsize=15)
        plt.title(tt,fontsize=20)
        plt.tick_params(labelsize=15)
        
        ax = plt.gca() 
        ax.invert_yaxis()
            
        return  kp, des
    
    
    def get_good_match(des1, des2,KG=0.75):
#        bf = cv2.BFMatcher()
#        matches = bf.knnMatch(des1, des2, k=2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)# or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m, n in matches:
            if m.distance < KG * n.distance:
                good.append(m)
        return good

    kp1, des1 = sift_kp(img1, 'GST_TiO', mask1)
    kp2, des2 = sift_kp(img2, 'HMI_Continuum', mask2)

    goodMatch = get_good_match(des1, des2,KG=KG)
    if len(goodMatch) > 1:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    else:
        print('SORRY ! I cannot do it')
        img1Out=None
        img2Out='bad'
        img2Out0='bad'
        tform='bad'
        status='bad'
        src='bad'
        dst='bad'
        return img1Out, img2Out, img2Out0, tform, status,src,dst

    src = np.squeeze(ptsA)
    dst = np.squeeze(ptsB)
    
    import pandas as pd
    tmp=np.hstack((src,dst))
    newdata=pd.DataFrame(tmp,columns=['A','B','C','D'])
    s=newdata.drop_duplicates(subset=['A','B','C','D'],keep='first')
    tmp=np.array(s)
    src=tmp[:,:2]
    dst=tmp[:,2:]
    
    src2=src[:,::-1]
    dst2=dst[:,::-1]
#    model_robust =func()
#    model_robust.estimate(src, dst)
#    inlier_idxs=range(len(src))
    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), func, min_samples=2,
                                   residual_threshold=san, max_trials=500)
    outliers = inliers == False

    # visualize correspondence
    inlier_idxs = np.nonzero(inliers)[0]
    outlier_idxs = np.nonzero(outliers)[0]
    
    if debug == 1:
        print(len(inliers))
        fig, ax = plt.subplots(nrows=2, ncols=1)

        plt.gray()

        plot_matches(ax[0], img1, img2, src2, dst2,
                     np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
        ax[0].axis('off')
        ax[0].set_title('Correct correspondences')

        plot_matches(ax[1], img1, img2, src2, dst2,
                     np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
        ax[1].axis('off')
        ax[1].set_title('Faulty correspondences')

        plt.show()
        plt.pause(0.1)
        plt.draw()

    if scale is None:
        tform = SimilarityTransform(scale=model_robust.scale, rotation=model_robust.rotation,
                                    translation=model_robust.translation)
    else:    
        tform = SimilarityTransform(scale=1,rotation=model_robust.rotation,translation=model_robust.translation)
        
    img1Out = warp(img1, tform.inverse, output_shape=(img2.shape[0], img2.shape[1]))
    img2Out = warp(img2Org, tform, output_shape=(img1.shape[0], img1.shape[1]))
    img2Out0= warp(img3, tform, output_shape=(img1.shape[0], img1.shape[1]))
#
    img2Out = transform.resize(img2Out, (Hsize[0], Hsize[1]),mode='reflect')
    img2Out0= transform.resize(img2Out0, (Hsize[0], Hsize[1]),mode='reflect')

#    img2Out = transform.resize(img2Out, (Hsize[0], Hsize[1]),mode='reflect')
    status = (src[inlier_idxs], dst[inlier_idxs])

    return img1Out, img2Out, img2Out0, tform, status,src,dst


def fitswrite(fileout, im, header):
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head


def removelimb(im, center=None, RSUN=None):
    #  pip install polarTransform==1.0.1
    import polarTransform as pT
    from scipy import signal

    radiusSize, angleSize = 1024, 1840
    im = removenan(im)
    im2=im.copy()
    if center is None:
        T = (im.max() - im.min()) * 0.2 + im.min()
        arr = (im > T)
        import scipy.ndimage.morphology as snm
        arr=snm.binary_fill_holes(arr)
#        im2=(im-T)*arr
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        xc = (X * arr).astype(float).sum() / (arr*1).sum()
        yc = (Y * arr).astype(float).sum() / (arr*1).sum()
        center = (xc, yc)
        RSUN = np.sqrt(arr.sum() / np.pi)

    Disk = np.int8(disk(im.shape[0], im.shape[1], RSUN * 0.95))
    impolar, Ptsetting = pT.convertToPolarImage(im, center, radiusSize=radiusSize, angleSize=angleSize)
    profile = np.median(impolar, axis=0)
    profile = signal.savgol_filter(profile, 11, 3)
#    Z = profile.reshape(-1, 1).repeat(impolar.shape[1], axis=1)
    Z = profile.reshape(-1, 1).T.repeat(impolar.shape[0], axis=0)
    Z=Ptsetting.convertToCartesianImage(Z)
    im2 = removenan(im / Z)-1
    im2 = im2 * Disk
    im = removenan(im-Z)
    im= im*Disk
    return im, center, RSUN, Disk,im2,Z


def imnorm(im, mx=0, mi=0):
    #   图像最大最小归一化 0-1
    if mx != 0 and mi != 0:
        pass
    else:
        mi, mx = np.min(im), np.max(im)

    im2 = removenan((im - mi) / (mx - mi))

    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0

    return im2


def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def showim(im):
    mi = np.max([im.min(), im.mean() - 3 * im.std()])
    mx = np.min([im.max(), im.mean() + 3 * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx)
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap='gray')

    return


def zscore2(im):
    im = (im - im.mean()) / im.std()
    return im


def disk(M, N, r0):
    X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)), np.linspace(-int(M / 2), int(M / 2) - 1, M))
    r = (X) ** 2 + (Y) ** 2
    r = (r ** 0.5)
    im = r < r0
    return im


def create_gif(images, gif_name, duration=1):
    import imageio
    frames = []
    # Read
    T = images.shape[2]
    for i in range(T):
        frames.append(np.uint8(imnorm(images[:, :, i]) * 255))
    #    # Save
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    return


def fixSDO(Sdo, hS,Disk=None):
    if Disk is None: Disk=1
    xc = hS['CRPIX1']
    yc = hS['CRPIX2']

    rot = hS['CROTA2']
    shift = [2048.5 - xc, 2048.5 - yc]

#    tform = SimilarityTransform(translation=shift)
#    Sdo = imnorm(removenan(Sdo)*Disk)
#    Sdo2 = warp(Sdo, tform, output_shape=(Sdo.shape[0], Sdo.shape[1]))
#    Sdo2 = rotate(Sdo2, -rot,mode='reflect')
    Sdo2=immove2(Sdo,shift[0],shift[1])
    Sdo2=imrotate(Sdo2,-rot)
    return Sdo2



def imrotate(im,rot):
    im2,para=array2img(im)
    im2=rotate(im2,rot,mode='constant')
    im2=img2array(im2,para)
    return im2

def immove2(im,dx=0,dy=0):
    im2,para=array2img(im)
    tform = SimilarityTransform(translation=(dx,dy))
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='reflect')
    im2=img2array(im2,para)
    return im2

def imresize(im,scale):
    im2,para=array2img(im)
    im2=rescale(im2,scale,mode='reflect')
    im2=img2array(im2,para)
    return im2

def array2img(im):
    Bzero=im.min()
    mx=im.max()
    Bscale=mx-Bzero
    im2=(im-Bzero)/Bscale
    para=(Bzero,Bscale)
    return im2,para

def img2array(im,para):
    im2=im*para[1]+para[0]
    return im2

def imcenterpix(im):
    X0=(im.shape[0]+1)//2
    Y0=(im.shape[1]+1)//2
    cen=(X0,Y0)
    return cen

def imtransform(im,scale=1,rot=0,translation=[0,0]):
    im2=im.copy()
    im2,para=array2img(im2)
    tform = SimilarityTransform(translation=translation)
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='reflect')

    im2=rotate(im2,rot,mode='reflect')
    im2=rescale(im2,scale,mode='reflect')


    im2=img2array(im2,para)
    return im2   
            
def fulldisk(im, scal,rot,center=None,size=[4096,4096],Disk=None):
    if Disk is None: Disk=np.ones(size)>0
    im=removenan(im)
    im2=im.copy()
    cen=np.array(imcenterpix(im)) 
    if center is None:
        T = (im.max() - im.min()) * 0.2 + im.min()
        arr = (im > T)
#        im2=(im-T)*arr
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        xc = (X * arr).astype(float).sum() / (arr*1).sum()
        yc = (Y * arr).astype(float).sum() / (arr*1).sum()
        center = [xc, yc]
        RSUN = np.sqrt(arr.sum() / np.pi)

    xc=center[0]
    yc=center[1]

#    im2=immove2(im2,-xc+cen[0],-yc+cen[1])
#    im2=imrotate(im2,rot)
#    im2 = imresize(im2, scal)
          
   
    shift=[-xc+cen[0],-yc+cen[1]]
    im2=imtransform(im2,scale=scal,rot=rot,translation=shift)
    
    cen=imcenterpix(im2)

    size2=(np.array(size)+1)//2
    im2=im2[cen[1]-size2[0]:cen[1]+size2[0],cen[0]-size2[1]:cen[0]+size2[1]]
    

#    im2=removenan(np.log(im2))
    mx=im2[Disk].max()
    mi=im2[Disk].min()
    im2 = imnorm(removenan(im2),mx=mx,mi=mi)
    return im2,center
def gaussfilter(im,sigma):
    im,para=array2img(im)
    im=filters.gaussian(im,sigma=sigma)
    im=img2array(im,para)
    return im