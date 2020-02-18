import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

matplotlib.rcParams['figure.figsize'] = 16,12


def process_frame(frame):
    frame = frame[100:, 170:480]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_bin = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    img_del = cv2.dilate(frame_bin, kernel, iterations=1)
    img_ero = cv2.erode(img_del, kernel, iterations=3)
    img_ero = cv2.dilate(img_ero, kernel, iterations=1)
    #     img_del = 255 - img_del
    plt.imshow(img_ero, 'gray')

    contours, hierarchy = cv2.findContours(img_ero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    img = frame.copy()
    count = 0
    for rect in rectangles:
        x, y, w, h = rect
        if (h > 20 and h < 55) and (w > 10 and w < 65) and (y > 40 and y < 45) and (x > 50 and x + w < 275):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img)

    return img, count


# frame = cv2.imread("../test5.jpg")
# process_frame(frame)
# plt.show()

def process_video(video_path):
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)
    frames = []
    full_count = 0
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        if not ret_val:
            break
        fr, count = process_frame(frame)
        full_count += count
        frames.append(fr)
    cap.release()
    return frames, full_count


def record(frames, video_id):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter("videos_conturs/video" + str(video_id) + "_conturs.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60,
                          size)
    for frame in frames:
        out.write(frame)
    out.release()


def count_all(to_record=False):
    with open('videos/res.txt', 'r') as file:
        reader = csv.reader(file)
        reader = [row for row in reader]
    errors = []
    for video_id in range(1, 11):
        frames, video_count = process_video("videos/video" + str(video_id) + ".mp4")
        if to_record:
            record(frames, video_id)
        print("Video " + str(video_id) + ": " + str(video_count) + ", actual: " + reader[video_id][1])
        errors.append(abs(eval(reader[video_id][1]) - video_count))

    print("MEAN: " + str(sum(errors)/10.0))


if __name__ == "__main__":
    count_all()
