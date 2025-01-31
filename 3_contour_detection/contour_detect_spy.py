import cv2
import matplotlib.pyplot as plt
import numpy as np


#resim ice aktarma
img  = cv2.imread("contour.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")



contours ,hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#retr ccomp bize resme ait ic ve dis contourlerin ikisini de bulmamiza yariyor
#cahin approx ise yatay dikey ve capraz boluimleri sikistirmamiiz sagliyor ve sadce uc noktalari birakiyor.



external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)



for i in range(len(contours)):
    #external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255,-1)
    else: #internal
        cv2.drawContours(internal_contour,contours, i, 255,-1)
        
plt.figure(), plt.imshow(external_contour,cmap="gray"),plt.axis("off")
plt.figure(), plt.imshow(internal_contour,cmap="gray"),plt.axis("off")












































