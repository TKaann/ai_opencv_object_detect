import cv2
import matplotlib.pyplot as plt 
import numpy as np

#oncelikle resmimizi siyah beyaza ceviriyoruz.
#resmi ice aktariyoruz.
img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#simdi burada suyun da girintilerini cikintilarini tspit ediyor. yani bizim isimize yaramayanlari da aliyor. bunlar icin ayar cekicez.



#threshold degerlerini degistiricez ve bluring islemi yapicaz.
#ilk olarak threshold degerlerini degistirelim.

#resmimizin medianini ogrenelim.
med_val = np.median(img)
print(med_val)

#simdi alt ve ust threshold belirliycez. genellikle literaturde bunlar kullaniliyor.
low = int(max(0, (1 - 0.33)*med_val))    #burada yuze 67 siniz aliyoruz low un
high = int(min(255,(1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#cok az da olsa bi azalma var ama istedigimiz gibi degil suanda.


#simdi tum resme bluring islemi yapalim ve sonuca bakalarim


blurred_img = cv2.blur(img, ksize=(5,5))
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")


med_val = np.median(blurred_img)
print(med_val)


low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255,(1 + 0.33)*med_val))

print(low)
print(high)


edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")


#burada resimdeki nesnelerin tespitinden zyade kenar tespiti yapacagimiz icin ksize boyutumuzun 5 e 5 olmasi kenarlari tespit etmemizde ise yaradi.


























































