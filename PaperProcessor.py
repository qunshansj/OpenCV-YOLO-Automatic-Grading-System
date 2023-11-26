python


class PaperProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def resize_image(self, height=800):
        """调整图像大小"""
        h, w = self.image.shape[:2]
        ratio = height / float(h)
        new_dim = (int(w * ratio), height)
        resized = cv2.resize(self.image, new_dim, interpolation=cv2.INTER_AREA)
        return resized

    def thresholding(self):
        """图像阈值化"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def correct_skew(self, image):
        """图像倾斜矫正"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def remove_handwritten(self):
        """手写字符图像去除"""
        # 这里简单使用阈值化和形态学处理来去除一些手写字符
        thresh = self.thresholding()
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        return opening

    def morphological_processing(self):
        """形态学处理"""
        # 这里以膨胀为例
        thresh = self.thresholding()
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        return dilated

    def denoise(self):
        """图像去噪"""
        # 使用高斯滤波进行去噪
   
   ......
