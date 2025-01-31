import cv2
import os


files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
    
print(img_path_list)



# hog tanımlayıcısı
hog = cv2.HOGDescriptor()
# tanımlayıcıa SVM ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
         
    cv2.imshow("Yaya: ",image)
    while True:
        key = cv2.waitKey(1) or 0xFF
        if key == ord("e"):  # E tuşuna basıldığında pencereyi kapat ve bir sonraki resme geç
            cv2.destroyAllWindows()
            break
        elif key == ord("q"):  # Q tuşuna basıldığında bir sonraki resme geç
            break

cv2.destroyAllWindows()  # Tüm pencereleri kapat





























































