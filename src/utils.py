import os
import cv2 as cv
import numpy as np


def preprocess_gradient(gray_img):
    # Apply morphological gradient
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    gradient = cv.morphologyEx(gray_img, cv.MORPH_GRADIENT, kernel)
    # Combine with original
    combined = cv.addWeighted(gray_img, 0.5, gradient, 0.5, 0)
    # Histogram equalization
    combined = combined.astype(np.uint8)
    return cv.equalizeHist(combined)


"""

Source - https://stackoverflow.com/a/65314036
Posted by Leonardo Mariga
Retrieved 2026-02-11, License - CC BY-SA 4.0

"""


def crop_square(img, size, interpolation=cv.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv.resize(crop_img, (size, size), interpolation=interpolation)
    resized = resized.astype(np.uint8)

    return resized


def detect_iris(eye_gray):
    """
    Robust iris detection for CASIA images
    Returns normalized iris and circle parameters.
    """
    h, w = eye_gray.shape

    # -------------------------------------------------
    # 1. Focus on central region (remove eyelids)
    # -------------------------------------------------
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    roi = eye_gray[y1:y2, x1:x2]

    # -------------------------------------------------
    # 2. Preprocessing
    # -------------------------------------------------
    roi = cv.equalizeHist(roi)
    #roi_blur = cv.GaussianBlur(roi, (5, 5), 1)
    roi_blur = cv.GaussianBlur(roi, (9, 9), 2)

    # -------------------------------------------------
    # 3. Hough Circle (CASIA tuned)
    # -------------------------------------------------
    min_r = int(min(roi.shape) * 0.15)
    max_r = int(min(roi.shape) * 0.45)

    circles = cv.HoughCircles(
        roi_blur,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        #original dp=1.2,
        minDist=roi.shape[0] // 2,
        param1=50,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")

    # -------------------------------------------------
    # 4. Select darkest circle (iris/pupil region)
    # -------------------------------------------------
    best_circle = None
    min_intensity = 1e12

    for (x, y, r) in circles:

        if x-r < 0 or y-r < 0 or x+r > roi.shape[1] or y+r > roi.shape[0]:
            continue

        patch = roi[y-r:y+r, x-r:x+r]
        mean_val = np.mean(patch)

        if mean_val < min_intensity:
            min_intensity = mean_val
            best_circle = (x, y, r)

    if best_circle is None:
        return None

    # Convert coordinates back to original eye image
    x, y, r = best_circle
    x += x1
    y += y1

    # -------------------------------------------------
    # 5. Crop normalized iris
    # -------------------------------------------------
    if x-r < 0 or y-r < 0 or x+r > w or y+r > h:
        return None

    iris = eye_gray[y-r:y+r, x-r:x+r]
    iris = crop_square(img=iris, size=80)

    #iris = cv.resize(iris, (80, 80))

    return iris, (x, y, r)



def detect_faces(gradient_img, detector, gray_img, face_dir, img_name):
    faces = detector.detectMultiScale(gradient_img, scaleFactor=1.0001,minNeighbors=6,minSize=(200, 200))
    
    face_id = 0
    eye_id = 0

    #img_saving = {0:'left_eye', 1: 'right_eye'}

    os.makedirs(face_dir, exist_ok=True)

    if len(faces) == 0:
        return []

    # Keep largest face (CASIA assumption)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    faces = [faces[0]]

    for (x, y, fw, fh) in faces:
        # ---- Save face ----
        face_gray = gray_img[y:y+fh, x:x+fw]
        face_path = os.path.join(face_dir, f"{img_name}_face_{face_id}.png")
        cv.imwrite(face_path, face_gray)
        face_id += 1

        # ---- Eye detection inside face ----
        eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(20, 20))
        # Sort and keep two eyes

        eyes = sorted(eyes, key=lambda e: e[0])

        for eye_id, (ex, ey, ew, eh) in enumerate (eyes):
            eye_gray = face_gray[ey:ey+eh, ex:ex+ew]
            eye_label = 'left_eye' if eye_id == 0 else 'right_eye'
            #eye_path = os.path.join(face_dir, f"{img_name}_eye_{eye_label}.png")
            eye_path = os.path.join(face_dir, f"{img_name}_{eye_label}.png")
            cv.imwrite(eye_path, eye_gray)
            #eye_id += 1

    return faces

# initializing haarcascade eyes detector
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')



def detect_save_iris2(faces, eye_cascade, gray_img, img_name, iris_dir):

    os.makedirs(iris_dir, exist_ok=True)
    detected_irises = []
    face_id = 0
    iris_id = 0
    eye_id = 0

    for (x, y, fw, fh) in faces:
        
        face_gray = gray_img[y:y+fh, x:x+fw]
        face_id += 1

        eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(20, 20))
        # Sort and keep two eyes

        eyes = sorted(eyes, key=lambda e: e[0])

        for (ex, ey, ew, eh) in eyes:

            eye_gray = face_gray[ey:ey+eh, ex:ex+ew]
            eye_id += 1

            h, w = eye_gray.shape

            y1 = int(h * 0.15)
            y2 = int(h * 0.85)
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)

            roi = eye_gray[y1:y2, x1:x2]
            roi = cv.equalizeHist(roi)
            roi_blur = cv.GaussianBlur(roi, (9, 9), 2)

            min_r = int(min(roi.shape) * 0.15)
            max_r = int(min(roi.shape) * 0.45)

            circles = cv.HoughCircles(
                roi_blur,
                cv.HOUGH_GRADIENT,
                dp=1.2,
                minDist=roi.shape[0] // 2,
                param1=50,
                param2=20,
                minRadius=min_r,
                maxRadius=max_r
                )

            if circles is None:
                continue

            circles = np.round(circles[0]).astype(int)

            best_circle = None
            min_intensity = np.inf

            for (cx, cy, r) in circles:

                if cx-r < 0 or cy-r < 0 or cx+r > roi.shape[1] or cy+r > roi.shape[0]:
                    continue

                patch = roi[cy-r:cy+r, cx-r:cx+r]
                mean_val = np.mean(patch)

                if mean_val < min_intensity:
                    min_intensity = mean_val
                    best_circle = (cx, cy, r)

                if best_circle is None:
                    continue

                cx, cy, r = best_circle
                x = cx + x1
                y = cy + y1

                if x-r < 0 or y-r < 0 or x+r > w or y+r > h:
                    continue

                iris = eye_gray[y-r:y+r, x-r:x+r]
                iris_img = crop_square(iris, size=512)

                cv.imwrite(
                    os.path.join(iris_dir, f"{img_name}_iris_{iris_id}.jpg"),
                    iris_img
                )
                detected_irises.append((iris_img, (x, y, r)))
                iris_id += 1

    return detected_irises



# def detect_save_iris(faces, gray_img, img_name, iris_dir):

#     os.makedirs(iris_dir, exist_ok=True)

#     detected_irises = []
#     iris_id = 0

#     for (fx, fy, fw, fh) in faces:

#         face_gray = gray_img[fy:fy+fh, fx:fx+fw]

#         eyes = eye_cascade.detectMultiScale(
#             face_gray,
#             scaleFactor=1.05,
#             minNeighbors=4,
#             minSize=(20, 20)
#         )

#         if len(eyes) == 0:
#             continue

#         eyes = sorted(eyes, key=lambda e: e[0])

#         for (ex, ey, ew, eh) in eyes:

#             eye_gray = face_gray[ey:ey+eh, ex:ex+ew]

#             h, w = eye_gray.shape

#             y1 = int(h * 0.15)
#             y2 = int(h * 0.85)
#             x1 = int(w * 0.15)
#             x2 = int(w * 0.85)

#             roi = eye_gray[y1:y2, x1:x2]
#             roi = cv.equalizeHist(roi)
#             roi_blur = cv.GaussianBlur(roi, (9, 9), 2)

#             min_r = int(min(roi.shape) * 0.15)
#             max_r = int(min(roi.shape) * 0.45)

#             circles = cv.HoughCircles(
#                 roi_blur,
#                 cv.HOUGH_GRADIENT,
#                 dp=1.2,
#                 minDist=roi.shape[0] // 2,
#                 param1=50,
#                 param2=20,
#                 minRadius=min_r,
#                 maxRadius=max_r
#             )

#             if circles is None:
#                 continue

#             circles = np.round(circles[0]).astype(int)

#             best_circle = None
#             min_intensity = np.inf

#             for (cx, cy, r) in circles:

#                 if cx-r < 0 or cy-r < 0 or cx+r > roi.shape[1] or cy+r > roi.shape[0]:
#                     continue

#                 patch = roi[cy-r:cy+r, cx-r:cx+r]
#                 mean_val = np.mean(patch)

#                 if mean_val < min_intensity:
#                     min_intensity = mean_val
#                     best_circle = (cx, cy, r)

#             if best_circle is None:
#                 continue

#             cx, cy, r = best_circle
#             x = cx + x1
#             y = cy + y1

#             if x-r < 0 or y-r < 0 or x+r > w or y+r > h:
#                 continue

#             iris = eye_gray[y-r:y+r, x-r:x+r]
#             iris_img = crop_square(iris, size=512)

#             cv.imwrite(
#                 os.path.join(iris_dir, f"{img_name}_iris_{iris_id}.jpg"),
#                 iris_img
#             )

#             detected_irises.append((iris_img, (x, y, r)))
#             iris_id += 1

#     return detected_irises

# def detect_save_iris(faces, gray_img, img_name, iris_dir):
#     """
#     Robust iris detection for CASIA images
#     Returns normalized iris and circle parameters.
#     """
#     os.makedirs(iris_dir, exist_ok=True)

#     face_id = 0
#     eye_id = 0
#     iris_id = 0

#     for (x, y, fw, fh) in faces:
#         # ---- Save face ----
#         face_gray = gray_img[y:y+fh, x:x+fw]
#         face_id += 1

#         # ---- Eye detection inside face ----
#         eyes = eye_cascade.detectMultiScale(
#             face_gray,
#             scaleFactor=1.05,
#             minNeighbors=4,
#             minSize=(20, 20)
#         )

#         eyes = sorted(eyes, key=lambda e: e[0])

#         for (ex, ey, ew, eh) in eyes:
#             eye_gray = face_gray[ey:ey+eh, ex:ex+ew]
#             eye_id += 1

#             # ---------------------------
#             # Iris detection (NEW)
#             # ---------------------------

#             h, w = eye_gray.shape

#             # -------------------------------------------------
#             # 1. Focus on central region (remove eyelids)
#             # -------------------------------------------------
#             y1 = int(h * 0.15)
#             y2 = int(h * 0.85)
#             x1 = int(w * 0.15)
#             x2 = int(w * 0.85)

#             roi = eye_gray[y1:y2, x1:x2]

#             # -------------------------------------------------
#             # 2. Preprocessing
#             # -------------------------------------------------
#             roi = cv.equalizeHist(roi)
#             #roi_blur = cv.GaussianBlur(roi, (5, 5), 1)
#             roi_blur = cv.GaussianBlur(roi, (9, 9), 2)

#             # -------------------------------------------------
#             # 3. Hough Circle (CASIA tuned)
#             # -------------------------------------------------
#             min_r = int(min(roi.shape) * 0.15)
#             max_r = int(min(roi.shape) * 0.45)

#             circles = cv.HoughCircles(
#                 roi_blur,
#                 cv.HOUGH_GRADIENT,
#                 dp=1.2,
#                 #original dp=1.2,
#                 minDist=roi.shape[0] // 2,
#                 param1=50,
#                 param2=20,
#                 minRadius=min_r,
#                 maxRadius=max_r
#             )

#             if circles is None:
#                 return None

#             circles = np.round(circles[0, :]).astype("int")

#             # -------------------------------------------------
#             # 4. Select darkest circle (iris/pupil region)
#             # -------------------------------------------------
#             best_circle = None
#             min_intensity = 1e12

#             for (x, y, r) in circles:

#                 if x-r < 0 or y-r < 0 or x+r > roi.shape[1] or y+r > roi.shape[0]:
#                     continue

#                 patch = roi[y-r:y+r, x-r:x+r]
#                 mean_val = np.mean(patch)

#                 if mean_val < min_intensity:
#                     min_intensity = mean_val
#                     best_circle = (x, y, r)

#             if best_circle is None:
#                 return None

#             # Convert coordinates back to original eye image
#             x, y, r = best_circle
#             x += x1
#             y += y1

#             # -------------------------------------------------
#             # 5. Crop normalized iris
#             # -------------------------------------------------
#             if x-r < 0 or y-r < 0 or x+r > w or y+r > h:
#                 return None

#             iris = eye_gray[y-r:y+r, x-r:x+r]

#             iris_img = crop_square(img=iris, size=512)

#             # Save iris
#             cv.imwrite(os.path.join(iris_dir, f"{img_name}_iris_{iris_id}.jpg"), iris_img)
#             iris_id += 1

#             return iris, (x, y, r)