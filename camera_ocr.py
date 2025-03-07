import cv2
import numpy as np
from paddleocr import PaddleOCR 

ocr = PaddleOCR(lang='en')

GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

cap = cv2.VideoCapture(0)

def process_frame(frame):
    license_plate = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        # Tính chu vi contour
        perimeter = cv2.arcLength(contour, True)
        # Aproximates a polygonal curve với độ chính xác 2% chu vi
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # Kiểm tra nếu contour có 4 cạnh (hình chữ nhật)
        if len(approx) == 4:
            # Sắp xếp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left

            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            # Lấy ma trận biến đổi phối cảnh và áp dụng
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))

            # Lưu biển số đã biến đổi
            license_plate = warped
            break
    if license_plate is None:
        print('Không tìm thấy biển số xe')
    else:
        cv2.imwrite("license.jpg", license_plate)
        result = ocr.ocr("license.jpg", cls=True)
        plate_numbers = []
        if result and any(result): 
            for line in result:
                print(line)
                if line:  
                    for word in line:
                        box, (text, confidence) = np.array(word[0], dtype=np.int32), word[1]
                        if confidence > 0.85:
                            plate_numbers.append(text)

                            x_min, y_min = np.min(box, axis=0)
                            x_max, y_max = np.max(box, axis=0)
                            plate_image = frame[y_min:y_max, x_min:x_max]

                            cv2.imwrite("plate.jpg", plate_image)

                            cv2.imshow("Biển số", plate_image)

                            cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    

        if plate_numbers:
            print(f"📸 Biển số xe: {' '.join(plate_numbers)}")
        else:
            print("❌ Không tìm thấy biển số trong khung hình")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể lấy hình ảnh từ camera")
        break
    process_frame(frame)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
