import cv2
from cv2 import VideoCapture, namedWindow, imshow, waitKey


#video_uri = "http://192.168.1.9:8080/video"    # IP Webcam App
#video_uri = 'v4l2://'  # Droidcam App as USB

class Window(object):
    def __init__(self, title="Video Stream"):
        ''' Uses OpenCV 2.3.1 method of accessing camera '''
        self.title = title
        self.cap = VideoCapture(0)
        self.prev = self.get_frame()
        self.frame = self.get_frame()
        namedWindow(title, 1) 

    def get_frame(self):
        success, frame = self.cap.read()
        return self.to_grayscale(frame) if success else False

    def to_grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def optical_flow(self): # FIXME
        flow = CreateImage(GetSize(frame), 32, 2)
        CalcOpticalFlowFarneback(self.prev, self.frame, flow, # takes 0.05s
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        return flow

    def disparity(self):
        if self.left is None or self.right is None:
            print "Capture left and right images using 'l' and 'r' keys before running disparity"
            return None

        hl, wl = self.left.shape
        hr, wr = self.right.shape
        disp_left  = cv2.cv.CreateMat(hl, wl, cv2.cv.CV_16S)
        disp_right = cv2.cv.CreateMat(hr, wr, cv2.cv.CV_16S)
        state = cv2.cv.CreateStereoGCState(16,2)
        # running the graph-cut algorithm
        from cv2.cv import fromarray
        cv2.cv.FindStereoCorrespondenceGC(fromarray(self.left), fromarray(self.right), disp_left, disp_right, state)
        cv2.cv.Save( "left.png", disp_left) # save the map
        cv2.cv.Save( "right.pgm", disp_right) # save the map

    def mainloop(self):
        while True:
            self.prev = self.frame
            self.frame = self.get_frame()
            sift(self.frame) # takes ~0.14s!
            imshow(self.title, self.frame)
            k = waitKey(10)
            if k == -1:
                pass
            elif chr(k) == 'l':
                self.left = self.frame
            elif chr(k) == 'r':
                self.right = self.frame
            elif chr(k) == 'd':
                self.disparity()
            elif k == 27:
                break


def sift(img):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)
    for pt in skp:
        cv2.circle(img, tuple(map(int, pt.pt)), 1, 255, -1)


if __name__ == "__main__":
    w = Window()
    w.mainloop()
