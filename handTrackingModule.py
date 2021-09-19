import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands  # for getting the value of different landmarks on the hand
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)  # for detection of hands
        self.mpDraw = mp.solutions.drawing_utils  # for drawing lines b/w all 21 points in hand detection

    def findHand(self, img, draw = True):
        imgRGB = cv2.cvtColor(img,
                              cv2.COLOR_BGR2RGB)  # we will convert bgr to rgb because our class usus only rgb images
        self.results = self.hands.process(imgRGB)  # it will process the frame for us

        # print(results.multi_hand_landmarks) # to check i hands are detected or not

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True ):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # it gives the ratio of image
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255),cv2.FILLED)  # IT WILL DRAW A CIRCLE FOR ID = 0 ON THE IMG

        return lmList


# static_image_mode = true makes the camera to continuously detect which will make the program very slow
# therefore we will keep it false in which it will track and detect on the basis of confidence level

def main():
    # for fps
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPosition(img)
        if(len(lmList) != 0):
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()