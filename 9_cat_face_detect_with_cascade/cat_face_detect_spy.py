import cv2
import os



#artik resimleri iceriye aktarirken tek tek degil de tek seferde birden fazla resim akiaricaz.
#dongu ile resimleri tek seferde aktarma islemi yapicaz.

files = os.listdir()
print(files)
img_path_list = []
for f in files:
    if f.endswith(".jpg"): 
        img_path_list.append(f)
print(img_path_list)


for j in img_path_list:
    print(j)
    image = cv2.imread(j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    #buradaki scaleFactor resim uyerinde ne kadar zoom yapacagini belirledigimiz bir parametredir.
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    for (i, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
        
    cv2.imshow(j, image)
    while True:
        key = cv2.waitKey(1) or 0xFF
        if key == ord("e"):  # E tuşuna basıldığında pencereyi kapat ve bir sonraki resme geç
            cv2.destroyAllWindows()
            break
        elif key == ord("q"):  # Q tuşuna basıldığında bir sonraki resme geç
            break

cv2.destroyAllWindows()  # Tüm pencereleri kapat



























































