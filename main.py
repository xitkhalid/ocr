import cv2
import easyocr
import pytesseract
import numpy as np
import os

class SmartOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.tess_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

    def preprocess_for_tesseract(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # থ্রেশহোল্ডিং
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # লাইন মুছে ফেলার চেষ্টা
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(thresh, (x1,y1), (x2,y2), (255,255,255), 2)
        
        # মরফোলজি দিয়ে নয়েজ মুছুন
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return morph

    def ocr_easyocr(self, img_path):
        try:
            result = self.reader.readtext(img_path, detail=0)
            if result and len(result) > 0:
                text = ''.join(result).strip()
                return text
        except Exception as e:
            print("[EasyOCR Error]", e)
        return None

    def ocr_tesseract(self, img_path):
        try:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(thresh, config=self.tess_config)
            return text.strip() if text.strip() else None
        except Exception as e:
            print("[Tesseract Error]", e)
        return None

    def ocr_tesseract_with_line_removal(self, img_path):
        try:
            processed = self.preprocess_for_tesseract(img_path)
            text = pytesseract.image_to_string(processed, config=self.tess_config)
            return text.strip() if text.strip() else None
        except Exception as e:
            print("[Tesseract + Line Removal Error]", e)
        return None

    def clean_text(self, text):
        # শুধু অক্ষর ও সংখ্যা রাখুন
        if not text:
            return ""
        cleaned = ''.join(c for c in text if c.isalnum())
        return cleaned

    def get_best_result(self, img_path):
        print(f"\n🔍 Processing: {img_path}")
        
        # 1. EasyOCR
        result = self.ocr_easyocr(img_path)
        if result:
            cleaned = self.clean_text(result)
            if len(cleaned) >= 3:  # কমপক্ষে 3 অক্ষর হতে হবে
                print(f"✅ EasyOCR Result: {cleaned}")
                return cleaned
        
        # 2. Tesseract (Basic)
        result = self.ocr_tesseract(img_path)
        if result:
            cleaned = self.clean_text(result)
            if len(cleaned) >= 3:
                print(f"✅ Tesseract Basic: {cleaned}")
                return cleaned
        
        # 3. Tesseract + Line Removal
        result = self.ocr_tesseract_with_line_removal(img_path)
        if result:
            cleaned = self.clean_text(result)
            if len(cleaned) >= 3:
                print(f"✅ Tesseract + Line Removal: {cleaned}")
                return cleaned
        
        # 4. Fallback: সব চেষ্টা ব্যর্থ হলে
        print("❌ All methods failed. Returning empty.")
        return ""

# 🚀 ব্যবহার করুন:
if __name__ == "__main__":
    ocr_system = SmartOCR()

    # আপনার ইমেজগুলোর পাথ (example)
    images = [
        "captcha1.jpg",  # 9LFU3
        "captcha2.jpg",  # TFUJJ
        "captcha3.jpg",  # aEHmA
    ]

    for img in images:
        if os.path.exists(img):
            final_text = ocr_system.get_best_result(img)
            print(f"🎯 Final Output for {img}: '{final_text}'")
        else:
            print(f"⚠️ Image not found: {img}")
