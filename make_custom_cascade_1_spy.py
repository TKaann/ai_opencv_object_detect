#simdi burada yapacagimiz sey kameramizi actigimizda kameramizin cektigi nesnenin her 5 frame de 1 tanesini kaydetmesi ve
#kaydettikleri resimleri yeni bir klasor icinde depolamasi, kaydettigi her resime ne kadar kaydettiyse o sayinin ismini vermesi,
#bunu taniyacagimiz verinin toplanmasi icin yapiyoruz. 
#ayrica bir de tanimak istedigimiz nesneden farkli bir nesne de cekiyoruz ki ogrenme islemini yapabilelim
#bu rastgele bir nesne olabilir.



import cv2
import os



# resim depo klasörü
path = "images"

# resim boyutu
imgWidth = 180
imgHeight = 120



# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path+str(countFolder))

saveDataFunc()

count = 0
countSave = 0

while True:
    
    success, img = cap.read()
    
    if success:
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave += 1
            print(countSave)
        count += 1
        
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()



#simdi cascde icin gerekli programi indiriyoru cascade trainer 
# https://amin-ahmadi.com/cascade-trainer-gui/
































































