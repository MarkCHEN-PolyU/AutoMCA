import time
import numpy as np
import win32api
import math
from mss import mss
import PySimpleGUI as sg
import os
from ctypes import windll

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp

# Make program aware of DPI scaling
user32 = windll.user32
user32.SetProcessDPIAware()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_utils = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
        model_complexity=2,
        enable_segmentation=True,
        smooth_segmentation=True
        )


def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
        else:
            is_reading, img = camera.read()
            if is_reading:
                working_ports.append(dev_port)
        dev_port += 1
    return working_ports


def angle_between_two_vectors(Vec1, Vec2):
    x1 = Vec1[0]
    y1 = Vec1[1]
    x2 = Vec2[0]
    y2 = Vec2[1]
    angle = math.acos((x1 * x2 + y1 * y2) / (math.sqrt(x1 ** 2 + y1 ** 2) * math.sqrt(x2 ** 2 + y2 ** 2)))
    return angle


def frame_rate_calculation(t_0, t_1):
    fps = 1 / (t_1 - t_0)
    return fps


Ports = list_ports()
PythonFilePath = os.getcwd()

CaptureImageFolderPath = PythonFilePath + '\\Captured Images'
if not os.path.exists(CaptureImageFolderPath):
    os.mkdir(CaptureImageFolderPath)

# Place the camera on the users' Left of Right
CamDirection = 'Right'
CameraIndex = 0
isCamSelectionChanged = False

time_old = 0

# Factor converts Pixel to Real Distance
k_p2r = 1

# Make program aware of DPI scaling
user32 = windll.user32
user32.SetProcessDPIAware()

# Define the GUI
sg.theme('DarkAmber')
PanelLayout = [
    [sg.Titlebar(title="Cranial Angle Measurement", background_color='chocolate4')],
    [sg.Text('Camera Setting', font='Times 13 bold')],
    [sg.Button('Search Available Camera'), sg.Text("Camera Selection:"),
     sg.DropDown(Ports, 0, key='-CamIndex-', readonly=True)],
    [sg.Text('Camera positioned on the: '),
     sg.DropDown(['Right', 'Left'], 'Right', key='-CamViewDirection-', readonly=True)],
    [sg.Text('Length Calibration', font='Times 13 bold')],
    [sg.Text('Distance From Canthus to Tragus: '), sg.InputText(10, key='-D_C2T_Real-', size=5), sg.Text('cm')],
    [sg.Text('HSV Color Thresholding', font='Times 13 bold')],
    [sg.Text('Colour Capture'), sg.DropDown(['On', 'Off'], 'Off', key='-CameraCaptureState-', readonly=True)],
    [sg.Text('HSV-Hue: '), sg.Text('', key='-HueValue-')],
    [sg.Slider([0, 180], 60, orientation='horizontal', key='-H-', disable_number_display=True),
     sg.Text('±'), sg.Slider([0, 64], 20, orientation='horizontal', key='-Tolerance_H-', disable_number_display=True)],
    [sg.Text('HSV-Saturation Range: '), sg.Text('', key='-SaturationRange-')],
    [sg.Slider([0, 255], 50, orientation='horizontal', key='-S_low-', disable_number_display=True),
     sg.Text('~'), sg.Slider([0, 255], 255, orientation='horizontal', key='-S_high-', disable_number_display=True)],
    [sg.Text('HSV-Value Range: '), sg.Text('', key='-ValueRange-')],
    [sg.Slider([0, 255], 50, orientation='horizontal', key='-V_low-', disable_number_display=True),
     sg.Text('~'), sg.Slider([0, 255], 255, orientation='horizontal', key='-V_high-', disable_number_display=True)],
    [sg.Text('Morphological Filtering', font='Times 13 bold')],
    [sg.Text('Filter Size'), sg.Text('', key='-FilterSizeDisplay-')],
    [sg.Slider([1, 20], 5, orientation='horizontal', key='-FilterSize-', disable_number_display=True)],
    [sg.Text('Camera View', font='Times 13 bold')],
    [sg.Text('Frame Rate: '), sg.Text('0', key='-FPS-')],
    [sg.Image(filename='', key='-IMAGE-', )],
    [sg.Checkbox('MediaPipe Landmarks', key='-Landmarks-', default=True),
     sg.Checkbox('ROI', key='-ROI-', default=True),
     sg.Checkbox('Detected Markers', key='-Markers-', default=True)],
    [sg.Text('Data Display', font='Times 13 bold'),
     sg.Checkbox('', key='-DataDisplay-', font='Times 13 bold', default=True)],
    [sg.Text('CVA (deg):', font='Arial 12'), sg.Text(0, key='-CVA-', font='Arial 12', size=5),
     sg.Text('CRA (deg):', font='Arial 12'), sg.Text(0, key='-CRA-', font='Arial 12', size=5),
     sg.Text('FHD (cm):', font='Arial 12'), sg.Text(0, key='-FHD-', font='Arial 12', size=5)],
    [sg.Button('Capture Image')]
]
PanelWindow = sg.Window('Control Panel', PanelLayout, keep_on_top=True)

# Initial the Webcam
cap = cv2.VideoCapture(0)

# Frame Rate Array
Arr_FPS = []

while True:
    if isCamSelectionChanged:
        isCamSelectionChanged = False
        cap.release()
        pose.reset()
        cap = cv2.VideoCapture(CameraIndex)
    event, values = PanelWindow.read(timeout=0)
    if event == sg.WIN_CLOSED:
        cap.release()
        break

    # Update Hue Value
    PanelWindow['-HueValue-'].update(str(values['-H-']) + '±' + str(values['-Tolerance_H-']))

    # Camera Select
    if values['-CamIndex-'] != CameraIndex:
        CameraIndex = values['-CamIndex-']
        isCamSelectionChanged = True
        continue
    if event == 'Show Camera View':
        isCameraViewDisplayed = not isCameraViewDisplayed
    if event == 'Search Available Camera':
        cap.release()
        Ports.clear()
        Ports = list_ports()
        PanelWindow['-CamIndex-'].update(values=Ports)
        PanelWindow['-CamIndex-'].update(value=0)
        CameraIndex = 0
        cap = cv2.VideoCapture(CameraIndex)
        continue

    # Update Saturation Value
    if values['-S_low-'] >= values['-S_high-']:
        PanelWindow['-S_high-'].update(values['-S_low-'])
    if values['-S_high-'] <= values['-S_low-']:
        PanelWindow['-S_low-'].update(values['-S_high-'])
    PanelWindow['-SaturationRange-'].update(str(values['-S_low-']) + '~' + str(values['-S_high-']))

    # Update Value Value
    if values['-V_low-'] >= values['-V_high-']:
        PanelWindow['-V_high-'].update(values['-V_low-'])
    if values['-V_high-'] <= values['-V_low-']:
        PanelWindow['-V_low-'].update(values['-V_high-'])
    PanelWindow['-ValueRange-'].update(str(values['-V_low-']) + '~' + str(values['-V_high-']))

    # Color Capturing (Click Left & Right Mouse Button)
    if values['-CameraCaptureState-'] == 'On':
        if win32api.GetKeyState(0x01) < 0 and win32api.GetKeyState(0x02) < 0:
            x_mouse, y_mouse = win32api.GetCursorPos()
            # print(x_mouse, y_mouse)
            with mss() as sct:
                # mons = sct.monitors
                mon = {'left': x_mouse - 10, 'top': y_mouse - 10, 'width': 20, 'height': 20}
                pic = sct.grab(mon)
                input_image_np = np.array(pic)
                input_image_BGR = input_image_np
                input_image_hsv = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2HSV)
                ClickPoint_H = input_image_hsv[10, 10, 0]
                ClickPoint_S = input_image_hsv[10, 10, 1]
                ClickPoint_V = input_image_hsv[10, 10, 2]

                PanelWindow['-H-'].update(ClickPoint_H)
                PanelWindow['-S_low-'].update(ClickPoint_S - 30)
                PanelWindow['-S_high-'].update(ClickPoint_S + 30)
                PanelWindow['-V_low-'].update(ClickPoint_V - 30)
                PanelWindow['-V_high-'].update(ClickPoint_V + 30)

                sg.popup(str(ClickPoint_H) + ',' + str(ClickPoint_S) + ',' + str(ClickPoint_V), auto_close=True,
                         auto_close_duration=1, no_titlebar=True, keep_on_top=True)
                time.sleep(0.1)

    PanelWindow['-FilterSizeDisplay-'].update(values['-FilterSize-'])
    # Set Kernel Size
    kernel_size = int(values['-FilterSize-'])

    # Set Camera Direction
    CamDirection = values['-CamViewDirection-']

    # Read Image from cap
    success, input_image = cap.read()
    if not success:
        continue

    D_img = input_image.shape
    H_img = D_img[0]
    W_img = D_img[1]
    Y_ROI = 0
    H_ROI = H_img
    X_ROI = 0
    W_ROI = W_img

    output_image = input_image
    origin_image = input_image.copy()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    mp_results = pose.process(input_image)

    if mp_results.pose_landmarks is not None:
        x_nose = mp_results.pose_landmarks.landmark[0].x
        X_nose = round(x_nose * W_img)
        y_nose = mp_results.pose_landmarks.landmark[0].y
        Y_nose = round(y_nose * H_img)

        x_ear_left = mp_results.pose_landmarks.landmark[7].x
        x_ear_right = mp_results.pose_landmarks.landmark[8].x
        y_ear_left = mp_results.pose_landmarks.landmark[7].y
        y_ear_right = mp_results.pose_landmarks.landmark[8].y

        if x_nose > x_ear_left and x_nose > x_ear_right:
            PanelWindow['-CamViewDirection-'].update('Right')
        elif x_nose < x_ear_left and x_nose < x_ear_right:
            PanelWindow['-CamViewDirection-'].update('Left')

        if CamDirection == 'Right':
            y_ear = y_ear_right
            Y_ear = round(y_ear * H_img)
        elif CamDirection == 'Left':
            y_ear = y_ear_left
            Y_ear = round(y_ear * H_img)

        x_shoulder = (mp_results.pose_landmarks.landmark[11].x + mp_results.pose_landmarks.landmark[12].x)/2
        X_shoulder = round(x_shoulder * W_img)
        y_shoulder = (mp_results.pose_landmarks.landmark[11].y + mp_results.pose_landmarks.landmark[12].y)/2
        Y_shoulder = round(y_shoulder * H_img)

        delta_X_s2n = X_nose - X_shoulder
        delta_Y_s2n = Y_nose - Y_shoulder

        X_1 = min(round(X_nose + 0.25 * delta_X_s2n), W_img)
        X_1 = max(round(X_nose + 0.25 * delta_X_s2n), 0)

        Y_1 = max(round(Y_ear + 0.5 * delta_Y_s2n), 0)
        Y_1 = min(round(Y_ear + 0.5 * delta_Y_s2n), H_img)
        X_2 = min(round(X_shoulder - 1.0 * delta_X_s2n), W_img)
        X_2 = max(round(X_shoulder - 1.0 * delta_X_s2n), 0)

        Y_2 = min(Y_shoulder, H_img)
        Y_2 = max(Y_shoulder, 0)

        if values['-ROI-']:
            cv2.rectangle(output_image,
                          (X_1, Y_1),

                          (X_2, Y_2),
                          (255, 0, 0), 2)

        Y_ROI = min(Y_1, Y_2)
        H_ROI = abs(Y_2-Y_1)
        X_ROI = min(X_1, X_2)
        W_ROI = abs(X_2 - X_1)

        if values['-Landmarks-']:
            mp_drawing.draw_landmarks(
                output_image,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_utils.DrawingSpec([255, 0, 0], 3, 1),
                )

    try:
        input_image = input_image[Y_ROI:Y_ROI + H_ROI, X_ROI:X_ROI + W_ROI]
        input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
    except:
        continue

    mask_color_Hue = int(values['-H-'])

    lower_Hue = int(mask_color_Hue - values['-Tolerance_H-'])
    if lower_Hue <= 0:
        lower_Hue = 0

    higher_Hue = int(mask_color_Hue + values['-Tolerance_H-'])
    if higher_Hue >= 180:
        higher_Hue = 180

    lower_range = np.array([lower_Hue, int(values['-S_low-']), int(values['-V_low-'])])
    upper_range = np.array([higher_Hue, int(values['-S_high-']), int(values['-V_high-'])])

    # Set Mask
    mask = cv2.inRange(input_image_hsv, lower_range, upper_range)
    result = cv2.bitwise_and(input_image, input_image, mask=mask)

    # define the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # filter the image
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)

    # Color format conversion
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Find Contours
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the amount of contours
    n_contour = len(contours)

    # Define empty list to store the center position data
    arr_center_x = []
    arr_center_y = []

    for contour in contours:
        contour[:, 0, 0] = contour[:, 0, 0] + X_ROI
        contour[:, 0, 1] = contour[:, 0, 1] + Y_ROI
        x_array = contour[:, 0, 0]
        y_array = contour[:, 0, 1]
        x_mean = round(np.mean(x_array))
        y_mean = round(np.mean(y_array))
        # print(round(x_mean), round(y_mean))
        if values['-Markers-']:
            output_image = cv2.circle(output_image, (x_mean, y_mean), radius=5, color=(0, 0, 255), thickness=-1)
        arr_center_x.append(x_mean)
        arr_center_y.append(y_mean)

    if values['-Markers-']:
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    try:
        if CamDirection == 'Right':
            index_Canthus = arr_center_x.index(max(arr_center_x))
            index_C7 = arr_center_x.index(min(arr_center_x))
            Vec_Horizontal = [1, 0]
        elif CamDirection == 'Left':
            index_Canthus = arr_center_x.index(min(arr_center_x))
            index_C7 = arr_center_x.index(max(arr_center_x))
            Vec_Horizontal = [-1, 0]
        index_Tragus = 3 - index_C7 - index_Canthus

        # Draw Lines
        if values['-Markers-']:
            cv2.line(output_image, (arr_center_x[index_Tragus], arr_center_y[index_Tragus]),
                     (arr_center_x[index_Canthus], arr_center_y[index_Canthus]), color=(255, 0, 0), thickness=2)

            cv2.line(output_image, (arr_center_x[index_Tragus], arr_center_y[index_Tragus]),
                     (arr_center_x[index_C7], arr_center_y[index_C7]), color=(255, 0, 0), thickness=2)

        # Forward Head Distance (FHD) in Pixel
        FHD_Pixel = abs(arr_center_x[index_Tragus] - arr_center_x[index_C7])

        # Pixel Distance from Tragus to Canthus
        D_T2C_Pixel = math.sqrt((arr_center_x[index_Tragus] - arr_center_x[index_Canthus]) ** 2
                                + (arr_center_y[index_Tragus] - arr_center_y[index_Canthus]) ** 2)
        # Real Distance from Tragus to Canthus
        D_T2C_Real = float(values['-D_C2T_Real-'])  # cm
        k_p2r = D_T2C_Real / D_T2C_Pixel
        # print(FHD_Pixel, D_T2C_Pixel)

        # Forward Head Distance (FHD) in cm
        FHD_Real = FHD_Pixel * k_p2r

        # Cervical Angles
        CVA = math.atan((arr_center_y[index_C7]-arr_center_y[index_Tragus])/abs(arr_center_x[index_C7]-arr_center_x[index_Tragus]))
        CVA_DEG = CVA / math.pi * 180
        CRA = angle_between_two_vectors([(arr_center_x[index_Canthus] - arr_center_x[index_Tragus]),
                                         (arr_center_y[index_Canthus] - arr_center_y[index_Tragus])],
                                        [(arr_center_x[index_C7] - arr_center_x[index_Tragus]),
                                         (arr_center_y[index_C7] - arr_center_y[index_Tragus])])
        CRA_DEG = CRA / math.pi * 180

        if values['-Markers-']:
            cv2.putText(output_image, 'CVA: ' + str(round(CVA_DEG, 1)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(output_image, 'CRA: ' + str(round(CRA_DEG, 1)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(output_image, 'FHD(cm): ' + str(round(FHD_Real, 2)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 1, cv2.LINE_AA)

        if values['-DataDisplay-']:
            PanelWindow['-FHD-'].update(round(FHD_Real, 3))
            PanelWindow['-CVA-'].update(round(CVA_DEG, 1))
            PanelWindow['-CRA-'].update(round(CRA_DEG, 1))
        else:
            PanelWindow['-FHD-'].update(0)
            PanelWindow['-CVA-'].update(0)
            PanelWindow['-CRA-'].update(0)

    except:
        PanelWindow['-FHD-'].update(0)
        PanelWindow['-CVA-'].update(0)
        PanelWindow['-CRA-'].update(0)
        pass

    time_new = time.time()
    fps_current = frame_rate_calculation(time_old, time_new)
    Arr_FPS.append(fps_current)
    if len(Arr_FPS) > 10:
        Arr_FPS.pop(0)
    fps_ave = sum(Arr_FPS) / len(Arr_FPS)
    PanelWindow['-FPS-'].update(int(fps_ave))
    time_old = time_new
    ImgBytes = cv2.imencode('.png', output_image)[1].tobytes()
    PanelWindow['-IMAGE-'].update(data=ImgBytes)

    if event == 'Capture Image':
        TimeMark = str(time.localtime().tm_year) + "_" \
                   + str(time.localtime().tm_mon) + "_" \
                   + str(time.localtime().tm_mday) + " " \
                   + str(time.localtime().tm_hour) + "_" \
                   + str(time.localtime().tm_min) + "_" \
                   + str(time.localtime().tm_sec)

        CaptureImagePath_1 = CaptureImageFolderPath + '\\' + TimeMark + "_origin" + '.png'
        cv2.imwrite(CaptureImagePath_1, origin_image)
        CaptureImagePath = CaptureImageFolderPath + '\\' + TimeMark + "_AutoCAM" + '.png'
        cv2.imwrite(CaptureImagePath, output_image)
        sg.popup('Image Captured', font="Normal 30", no_titlebar=True, keep_on_top=True,
                 auto_close=True, auto_close_duration=1, button_type=sg.POPUP_BUTTONS_NO_BUTTONS)

cap.release()
