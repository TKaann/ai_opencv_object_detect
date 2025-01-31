import cv2
import matplotlib.pyplot as plt
import numpy as np


#resmi ice aktariyoruz.
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")


#corner detectiona geciyoruz.
#harris corner detection


dst = cv2.cornerHarris(img,blockSize = 2,ksize = 3, k= 0.04)
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off")


#simdi her kosedeki kareleri daha da belirginlestiricez.
#boyurlarini artirip rengini beyaz yaptik
dst = cv2.dilate(dst,None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off")


#simdi ise shi tomasi detection ile yapalim.
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)   
#buradaki 120 elde etmek istedigmiz kose sayisi. 0.01 ise quality lvl. 10 ise min distance yani 2 kose arasindaki min distance.
corners = np.int64(corners)


for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED)
    
plt.imshow(img)
plt.axis("off")











































