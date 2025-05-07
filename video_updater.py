import cv2
import pandas as pd
import numpy as np
import os


def crop_center_with_padding(image, center_x, center_y, size=100):
    half_size = size // 2
    h, w = image.shape[:2]

    # Számoljuk ki a kívánt kivágás koordinátáit
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size

    # Új, fehér háttér
    result = np.full((size, size, 3), 255, dtype=np.uint8)

    # Forrás kép érvényes része
    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    src_x2 = min(x2, w)
    src_y2 = min(y2, h)

    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return result  # nincs érvényes rész, visszaadunk csak fehéret

    # Cél hely az új képen
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Másolás
    result[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return result





def annotate_video_with_boxes(video_path: str, boxes_df: pd.DataFrame, output_path: str, measurements : pd.DataFrame, xcal=0.0, ycal=0.0):
    # Videó megnyitása
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    # Videó jellemzők
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    # Videó író inicializálása
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            # szem sebedséggének kiszámítása
    
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # végére értünk

        # Elszámolt idő másodpercben
        elapsed_time = frame_idx / fps

        ScreenId = "NONE"
        eye_speed = 0.0

        measurements_for_frame = measurements[measurements['SCN'] == frame_idx]
        if not measurements_for_frame.empty:
            fpogx = measurements_for_frame['FPOGX'].mean() * width
            fpogy = measurements_for_frame['FPOGY'].mean() * height
            fpogx = int(fpogx + xcal)
            fpogy = int(fpogy + ycal)
            image = crop_center_with_padding(frame, int(fpogx), int(fpogy), size=100)
            # ScreenId i need last value of the frame but only két tizedes jegyig
            # round to 2 decimal places
            ScreenId = str (measurements_for_frame['ScreenId'].values[-1])[:2]
            #ScreenId = str (measurements_for_frame['ScreenId'].values[-1])
            eye_speed = measurements_for_frame['eye_speed'].mean()  

            cv2.circle(frame, (int(fpogx), int(fpogy)), 10, (0, 0, 255), -1)
        # get 100pix100 pixel image and draw it to teh righ bottom coirnar of the frame

            # check if the image is empty
            if image.size != 0:
                frame[height-100:height, width-100:width] = image
        # Idő és képkockaszám felírása
        cv2.putText(frame, f'Time: {elapsed_time:.2f}s', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)
        cv2.putText(frame, f'Img: {frame_idx}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 128), 2)
        cv2.putText(frame, f'ScreenId: {ScreenId}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 128), 2)
        cv2.putText(frame, f'eye_speed: {eye_speed}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 128), 2)
        

        # Aktuális időpontban aktív négyzetek szűrése
        active_boxes = boxes_df[
            (boxes_df['Type'] == 1) &
            (boxes_df['Start'] <= elapsed_time) &
            (boxes_df['End'] >= elapsed_time)
        ]

        # Négyzetek kirajzolása
        for _, row in active_boxes.iterrows():
            bx1, by1, bx2, by2 = int(row['X1']*width), int(row['Y1']*height), int(row['X2']*width), int(row['Y2']*height)
            name = row['Name'] 
            #print(name)
 
            # Átlátszó négyzet rajzolása (csak keret, nem kitöltött)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            #cv2.imshow("Frame", frame)
            #cv2.waitKey(1)
            # Név megjelenítése kis keretben a négyzet fölött
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = bx1
            text_y = by1 - 10 if by1 - 10 > 20 else by1 + 20

            # háttér a név mögé (opcionális)
            cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
            
            cv2.putText(frame, name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Kocka írása kimeneti videóba
        #cv2.imshow("Frame", frame)
        #cv2.waitKey(1)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
