import cv2
import face_recognition

imgElon = face_recognition.load_image_file('_23AttendenseSystem\imagedata\Elon Musk.png')
# above line only load image same as cv2.imread('...')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon = cv2.resize(imgElon, (600, 500))

imgBill = face_recognition.load_image_file('_23AttendenseSystem\imagedata\Bill Gates.png')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)
imgBill = cv2.resize(imgBill, (600, 500))

# faceLoc = face_recognition.face_locations(imgElon) --> o/p=[(297, 810, 759, 348)]
faceLoc = face_recognition.face_locations(imgElon)[0] # (top, right, bottom, left)

# encodeElon = face_recognition.face_encodings(imgElon) --> o/p=Give array of single list in list 128 features are avaiable 
encodeElon = face_recognition.face_encodings(imgElon)[0] # we get list of 128 features
cv2.rectangle(imgElon, pt1=(faceLoc[3], faceLoc[0]), pt2=(faceLoc[1], faceLoc[2]), color=(0, 0, 255), thickness=2)

facelocBill = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill, pt1=(facelocBill[3], facelocBill[0]), pt2=(facelocBill[1], facelocBill[2]), color=(0, 0, 255), thickness=2)

results = face_recognition.compare_faces([encodeElon], encodeBill) # compare second face with list of give faces
# result = False means, not same person, result=True means, same person
facedis = face_recognition.face_distance([encodeElon], encodeBill)
print(results, facedis)
cv2.putText(imgBill, text=f"{results}, {round(facedis[0], 2)}", org=(50,50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 255), thickness=3)

cv2.imshow("ImageElon", imgElon)
cv2.imshow("ImageBill", imgBill)
cv2.waitKey(0)