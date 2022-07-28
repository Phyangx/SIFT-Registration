# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:35:59 2022

@author: Xu Yang
"""
# pip uninstall opencv-python
# pip install opencv-contrib-python==3.4.2.16

import numpy as np
from matplotlib import pyplot as plt
import FLAN_pho_sim_lib as sim
import glob
# import fit2d as fit2d

plt.close('all')

filedir='/media/xuyang/E/PSP/sample/'
filein=sorted(glob.glob(filedir+'bbso_tio_pcosr_20180702_162001.fts'))
fileout=filedir+'t1FOV_Calibrated_'+filein
lr,ud=0,0
K=1
KG=0.9
W=10

########################################## readfits
HighOrg,hH=sim.fitsread(filein)
Sdo,hS=sim.fitsread(filedir+'hmi.Ic_45s.20180702_162015_TAI.2.continuum.fits')

########################################## remove Sdo limb darkening
#Sdo=np.log(Sdo)
#sdob=fit2d.removebackground(Sdo)
#Sdo/=sdob
Sdo=sim.removenan(Sdo)
########################################## apply image mask
#Highscal=hH['CDELT1']
Highscal=0.06#0.06
Mask=np.zeros((HighOrg.shape[0],HighOrg.shape[1]))
Mask[110:-110,110:-110]=1
High=HighOrg.copy()*Mask
#High=High[::,::-1]
High=sim.removenan(High)
sdomask=Sdo>0

try:
    SDOscal=hS['CDELT1']
except:
    SDOscal=1  
xc = hS['CRPIX1']
yc = hS['CRPIX2'] 
RSUN=hS['RSUN_OBS']/SDOscal
# cen=[2048,2048]
cen=[xc,yc]
sc0= SDOscal/Highscal;
scale=None
Sdo,cen1,_,Disk,SDO,limb=sim.removelimb(Sdo,center=cen,RSUN=RSUN)
Sdo=sim.fixSDO(Sdo,hS,Disk=None)# fix HMI orientation and disk center
xc,yc=cen[0],cen[1]

############################################ down-sample the images
Sdo2,sHigh,sc= sim.per_data(Sdo,High,maskS=(sdomask>0),maskH=(High>0),sc0=sc0,lr=lr,ud=ud,K=K)
############################################ image registration
img1Out,img2Out,img2Out0,H,status, src, dst=sim.siftImageAlignment(sHigh,Sdo2,Sdo,High.shape,debug=1,mask1=np.int8(sHigh>0),mask2=sdomask>0,KG=KG,scale=scale)
############################################ registration information
print(Highscal*H.scale,H.rotation,H.translation)
Err=(H.residuals(status[0],status[1])**2)
Perr=np.sqrt(Err.sum()/(len(Err)-3))*SDOscal
print(Perr,len(Err))
############################################ make gif to show the registration results
frame=np.dstack((High,img2Out0))
sim.create_gif(frame,filein+'.mapx.gif')
########################################### update fits header
X0=(HighOrg.shape[0])/2-0.5
Y0=(HighOrg.shape[1])/2-0.5
X1=X0/sc
Y1=Y0/sc
X2=(sHigh.shape[0])/2-0.5
Y2=(sHigh.shape[1])/2-0.5

XY=np.squeeze(H((X1,Y1)))
hH['DATE_OBS']=hH['DATE-OBS']
hH['RSUN_OBS']=RSUN#hS['RSUN']
hH['CDELT1']=Highscal*H.scale*sc0/sc
hH['CDELT2']=Highscal*H.scale*sc0/sc
hH['CROTA1']=H.rotation/np.pi*180
hH['CROTA2']=H.rotation/np.pi*180
hH['CROTACN1']=(XY[0]-xc)*SDOscal
hH['CROTACN2']=(XY[1]-yc)*SDOscal
hH['CRPIX1']=X0-hH['CROTACN1']/hH['CDELT1']+1
hH['CRPIX2']=Y0-hH['CROTACN2']/hH['CDELT2']+1
hH['CUNIT1']='arcsec'
hH['CUNIT2']='arcsec'
hH['CRVAL1']=0
hH['CRVAL2']=0
hH['Creator']='Xu Yang'

############################################ write new fits
sim.fitswrite(fileout,High,hH)
