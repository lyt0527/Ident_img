import cv2
import numpy as np
from skimage.filters import gaussian
from PIL import Image
import pytesseract 
import math
import requests
from io import BytesIO
import multiprocessing as mp
from multiprocessing import Process,Lock,Array


class IDRecognise:
    def __init__(self):
        self = None
        
    def check_id_num(self, arr):
        if len(arr) < 15:
            return False
        area={"11":"北京","12":"天津","13":"河北","14":"山西","15":"内蒙古","21":"辽宁","22":"吉林","23":"黑龙江","31":"上海","32":"江苏","33":"浙江","34":"安徽","35":"福建","36":"江西","37":"山东","41":"河南","42":"湖北","43":"湖南","44":"广东","45":"广西","46":"海南","50":"重庆","51":"四川","52":"贵州","53":"云南","54":"西藏","61":"陕西","62":"甘肃","63":"青海","64":"宁夏","65":"新疆","71":"台湾","81":"香港","82":"澳门","91":"国外"}
        #print(((idcard)[0:2] in list(area)))
        if ((arr)[0:2] in list(area)) == False:
            return False
        #地区校验
        if(not area[(arr)[0:2]]):
            return False
        div = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check = [1, 0, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        temp = filter(lambda ch: ch in '0123456789xX', arr)
        arr = ''.join(list(temp))
        base = 0
        if 'x' in arr or 'X' in arr:
            base = 10
        arr = arr.replace('x', '0')
        arr = arr.replace('X', '0')
        if arr.isdigit() is True:
            if len(arr) == 18:
                d = int(arr[6])*1000 + int(arr[7])*100 + int(arr[8])*10 + int(arr[9])
                if d > 2100:
                    return False
                if d < 1900:
                    return False
                d = int(arr[10])*10 + int(arr[11])
                if (d < 1) or (d > 12):
                    return False
                total = 0
                for i in range(17):
                    total += int(arr[i]) * div[i]
                if int(arr[17]) + base == check[total % 11]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    
    def recogniseNumber(self, img): 
        config_dir = '--psm 7 -c tessedit_char_whitelist=0123456789xX'
        res = pytesseract.image_to_string(img, config = config_dir) 
        return res

    def WdivisionH(self, box):
        bound1 = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1] - box[1][1])**2)
        bound2 = math.sqrt((box[0][0]-box[3][0])**2 + (box[0][1] - box[3][1])**2)
        height = min(bound1, bound2)
        width = max(bound1, bound2)
        return width*1.0/height

    def deleteList(self, region):
        tempList = []
        L = sorted(region, key = self.WdivisionH, reverse=True)
        countN = 0
        for i in range(len(region)):
            if countN > 4:
                break
            tempList.append(L[i])
            countN = countN + 1
        return tempList

    def findTextArea(self, img): 
        region = []
        # 1. 查找轮廓
        contours, image = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area < 1500):
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # 计算高和宽
            bound1 = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1] - box[1][1])**2)
            bound2 = math.sqrt((box[0][0]-box[3][0])**2 + (box[0][1] - box[3][1])**2)
            height = min(bound1, bound2)
            width = max(bound1, bound2)
            if height < 20:
                continue
            if (height * 6 > width):
                continue
            if (height * 12 < width):
                continue
            region.append(box)
        region = self.deleteList(region)
        return region

    def subImage(self, img, Box): 
        orignal_W = math.ceil(np.sqrt((Box[0][0] - Box[3][0])**2 + (Box[0][1] - Box[3][1])**2)) 
        orignal_H = math.ceil(np.sqrt((Box[0][0] - Box[1][0])**2 + (Box[0][1] - Box[1][1])**2))
        pts1 = np.float32([Box[1], Box[2], Box[0], Box[3]])
        if(orignal_H > orignal_W):
            pts1 = np.float32([Box[2], Box[3], Box[1], Box[0]])
        pts2 = np.float32([[0,0],[max(orignal_H,orignal_W),0],[0,min(orignal_H,orignal_W)],[max(orignal_H,orignal_W),min(orignal_H,orignal_W)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        sub_img = cv2.warpPerspective(img, M, (max(orignal_H, orignal_W), min(orignal_H, orignal_W)))
        return sub_img

    def numberImage(self, img, threshold):
        #sub_img = cv2.Laplacian(sub_img, -1, ksize = 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
        sub_img = cv2.filter2D(img, -1, kernel=kernel)
        (_, sub_img) = cv2.threshold(sub_img, threshold, 255, cv2.THRESH_BINARY)
        #sub_img = cv2.medianBlur(sub_img, 3)
        return sub_img

    def grey_scale(self, image):
        rows, cols = image.shape
        flat_gray = image.reshape((cols * rows,)).tolist()
        A = min(flat_gray)
        B = max(flat_gray)
        if abs(A - B) == 0:
            return image
        C = np.mean(flat_gray)
        #print('A = %d,B = %d' %(A,B))
        output = np.uint8(255 / (B - A) * (image - A) + 0.5)
        return output

    def translationImage(self, image, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return image

    def diffImage(self, img, offset = 5):
        imgTemp = img.copy()
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        imgTemp = cv2.cvtColor(imgTemp.copy(), cv2.COLOR_BGR2GRAY)
        
        left = self.translationImage(img, -offset, 0)
        right = self.translationImage(img, offset, 0)
        up = self.translationImage(img, 0, -offset)
        down = self.translationImage(img, 0, offset)
        
        add_left = cv2.addWeighted(img, 0.5, left, 0.5, 0)
        add_right = cv2.addWeighted(img, 0.5, right, 0.5, 0)
        add_up = cv2.addWeighted(img, 0.5, up, 0.5, 0)
        add_down = cv2.addWeighted(img, 0.5, down, 0.5, 0)
        img = (img * 0.2 + add_left * 0.2 + add_right * 0.2+ add_up * 0.2+ add_down * 0.2)
        img = np.trunc(img)
        img = img.astype(int)
        img = Image.fromarray(img.astype(np.uint8))
        img = np.array(img)
        dst = cv2.absdiff(imgTemp, img)
        return dst

    def checkandrecognise(self, lock, sub_img, isStop):
        flag = False
        number = self.recogniseNumber(sub_img)
        #number.replace(" ", "")
        number = str(number)
        number = ''.join(number.split())
        #print(number)
        if self.check_id_num(number) == False:
            sub_img = np.rot90(sub_img, 1)
            sub_img = np.rot90(sub_img, 1)
        number = self.recogniseNumber(sub_img)
        number = str(number)
        number = ''.join(number.split())
        #加锁
        lock.acquire()
        if isStop[0] == 1:
            lock.release()
            return True
        if self.check_id_num(number):
            #print(number)
            #plt.imshow(sub_img)
            #plt.show()
            isStop[0] = 1
            #finalNumber = number
            for i in range(len(number)):
                isStop[1 + i] = int(number[i])
            #print(isStop[0])
            flag = True
        lock.release()
        return flag

    def recogniseImage(self, lock, dst, imgSrc, off, isStop):
        (_, dst) = cv2.threshold(dst, off, 255, cv2.THRESH_BINARY)
        dst = cv2.medianBlur(dst, 5)
        dst = cv2.medianBlur(dst, 5)
        dst = cv2.medianBlur(dst, 5)
        number, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        i=0
        for istat in stats:
            if istat[4]<100:
                if istat[3]>istat[4]:
                    r=istat[3]
                else:r=istat[4]
                cv2.rectangle(dst,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
            i=i+1
        array = [10, 15, 20, 25, 30, 5, 3]
        for i in range(len(array)):
            ele1 = cv2.getStructuringElement(cv2.MORPH_RECT, (array[i], array[i]))
            temp = cv2.dilate(dst, ele1, iterations=1)
            region = self.findTextArea(temp)
            #提取区域
            for i in range(len(region)):
                sub_img = self.subImage(imgSrc, region[i])
                #直接识别
                if self.checkandrecognise(lock, sub_img, isStop):
                    return True
                #锐化识别
                sub_img = self.USM(sub_img) * 255
                sub_img = sub_img.astype('uint8') 
                if self.checkandrecognise(lock, sub_img, isStop):
                    return True
                #灰度处理后识别
                sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
                sub_img = self.grey_scale(sub_img)
                if self.checkandrecognise(lock, sub_img, isStop):
                    return True
                #二值化后识别
                thresholdArray = [20, 40, 60, 80, 85, 100, 120]
                for j in range(len(thresholdArray)):
                    final = self.numberImage(sub_img, thresholdArray[j])
                    if self.checkandrecognise(lock, final, isStop):
                        return True
                    final = cv2.medianBlur(final, 5)
                    if self.checkandrecognise(lock, final, isStop):
                        return True
                    c, _ = cv2.findContours(final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    for i in range(len(c)):
                        area = cv2.contourArea(c[i])
                        if area < 25:
                            cv2.drawContours(final, [c[i]], 0, (255, 255, 255), -1)
                    if self.checkandrecognise(lock, final, isStop):
                        return True
        return False

    def Method3(self, img, offset = 5):
        imgSrc = img.copy()
        dst = self.diffImage(img, offset)
        rows, cols, _ = img.shape
        dst = dst[offset: rows - offset]
        dst = dst[:, offset: cols - offset]
        imgSrc = imgSrc[offset: rows - offset]
        imgSrc = imgSrc[:, offset: cols - offset]
        isStop = Array('i',range(19))
        isStop[0] = 0
        #print(isStop[:])
        #q = mp.Queue()
        lock = Lock()
        p1 = mp.Process(target=self.recogniseImage, args=(lock, dst.copy(), imgSrc, 5, isStop))
        p2 = mp.Process(target=self.recogniseImage, args=(lock, dst.copy(), imgSrc, 8, isStop))
        p3 = mp.Process(target=self.recogniseImage, args=(lock, dst.copy(), imgSrc, 3, isStop))
        p4 = mp.Process(target=self.recogniseImage, args=(lock, dst.copy(), imgSrc, 10, isStop))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        return isStop

    def USM(self, img):
        img = img * 1.0
        gauss_out = gaussian(img, sigma=5, multichannel=True)
        # alpha 0 - 5
        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img
        img_out = img_out/255.0
        # 饱和处理
        mask_1 = img_out  < 0 
        mask_2 = img_out  > 1
        img_out = img_out * (1-mask_1)
        img_out = img_out * (1-mask_2) + mask_2
        return img_out

    def myMethod(self, img):
        result = self.Method3(img)
        return result
    
    def getCardID(self, path):
        image = self.addressGetImage(path)
        result = self.myMethod(image)
        if result[0] == 0:
            return "未能找出身份证号"
        number = ""
        for i in range(1, len(result)):
            number = number + str(result[i])
        return number

    def addressGetImage(self, path):
        response = requests.get(path)
        image = Image.open(BytesIO(response.content))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # cap = cv2.VideoCapture(path)
        # ret = cap.isOpened()
        # ret, image = cap.read()
        # cap.release()
        image = np.array(image)
        (h, w, _) = image.shape
        if max(h, w)>1500:
            scale = self.getScale(image)
            size = (int(w * scale), int(h * scale))  
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    def getScale(self, img):
        (h, w, _) = img.shape
        d = max(h, w)
        return (1500.0 / d)

    def localFile(self, path):
        image = cv2.imread(path)
        (h, w, _) = image.shape
        if max(h, w)>1500:
            scale = self.getScale(image)
            size = (int(w * scale), int(h * scale))  
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image
