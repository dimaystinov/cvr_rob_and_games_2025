import cv2
import numpy as np
import mss

import pyautogui
import time
import threading

# Флаг для завершения программы
running = True
flag_allow_click_mouse = False
flag_allow_click_mouse_thread = False
number_of_shots = 6

# ROI для окна игры и для области, где бегают человечки
# (0, 0, 0, 0)
roi = (1378, 453, 974, 677) # region of interest
roi_2 = (32, 417, 731, 244)

x_global_click, y_global_click = 0, 0

from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Controller as KeyboardController, Key

def on_press(key):
    global running, flag_allow_click_mouse
    try:
        if key.char == '1':
            print("Клавиша 1 была нажата!")
            running = False
        elif key.char == '2':
            print(f"Клавиша 2 была нажата! {flag_allow_click_mouse} Можно ли кликать мышкой")
            flag_allow_click_mouse = not flag_allow_click_mouse
    except AttributeError:
        pass






from pynput.mouse import Button, Controller as MouseController

# Создаём контроллер для управления мышью
mouse_controller = MouseController()

def mouse_click(x=None, y=None, button='left'):
    try:
        if x is not None and y is not None:
            # Перемещение курсора в указанную позицию
            mouse_controller.position = (x, y)  # Установка позиции курсора
        # Определяем кнопку для клика
        if button == 'left':
            mouse_button = Button.left
        elif button == 'right':
            mouse_button = Button.right
        else:
            raise ValueError("Неподдерживаемая кнопка. Используйте 'left' или 'right'.")

        # Выполняем клик
        mouse_controller.click(mouse_button)
    except Exception as e:
        print(f"Произошла ошибка: {e}")



# Создаём контроллер для управления клавиатурой
keyboard_controller = KeyboardController()

def press_gap():
    global number_of_shots
    if number_of_shots <= 1:
        print("пробел")
        # Нажимаем и отпускаем пробел
        keyboard_controller.press(Key.space)  # Нажатие клавиши пробела
        time.sleep(0.05)  # Задержка (необязательно)
        keyboard_controller.release(Key.space)  # Отпускание клавиши пробела
        number_of_shots = 6





# Функция для создания ползунков
def create_trackbars():
    # Создаем окно с ползунками для настройки HSV-порогов
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda x: None)  # Low Hue
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)  # Low Saturation
    cv2.createTrackbar("L - V", "Trackbars", 200, 255, lambda x: None)  # Low Value
    cv2.createTrackbar("H - H", "Trackbars", 0, 179, lambda x: None)  # High Hue
    cv2.createTrackbar("H - S", "Trackbars", 0, 255, lambda x: None)  # High Saturation
    cv2.createTrackbar("H - V", "Trackbars", 229, 255, lambda x: None)  # High Value


# Функция для выбора ROI и последующего захвата области
def capture_and_filter_roi():
    global number_of_shots, roi, roi_2
    global x_global_click , y_global_click
    global flag_allow_click_mouse_thread
    global flag_allow_click_mouse


    with mss.mss() as sct:
        # Захват первого кадра
        monitor = sct.monitors[1]  # Первый монитор
        screenshot = sct.grab(monitor)

        # Преобразуем данные изображения в формат NumPy
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Отображаем первый кадр и позволяем пользователю выбрать ROI
        if sum(roi) == 0:
            roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)

            print('roi', roi)
            cv2.destroyAllWindows()

        # Координаты ROI: (x, y, w, h)
        x, y, w, h = roi

        if w == 0 or h == 0:
            print("ROI не выбран или имеет нулевой размер. Программа завершена.")
            return

        print(f"Выбранная ROI: x={x}, y={y}, width={w}, height={h}")

        # Создаем окно с ползунками
        create_trackbars()

        # Начинаем захват только выбранной области
        dict_number_last_x_coord = {}
        while running:
            # Проверяем, существует ли окно Trackbars
            if not cv2.getWindowProperty("Trackbars", cv2.WND_PROP_VISIBLE):
                # Если окно закрыто, пересоздаем его
                create_trackbars()

            # Захватываем только ROI
            screenshot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_game = img.copy()
            if sum(roi_2) != 0:
                x2, y2, w2, h2 = roi_2
                img = img[y2:y2 + h2, x2:x2 + w2, :]
            # Конвертируем изображение в HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height_img_game, width_img_game, _ = img.shape
            print(height_img_game, width_img_game)
            img_game_right_bottom_region = img_game[int(height_img_game * 0.9):, int(width_img_game * 0.9):,:]
            cv2.imshow('img_game_right_bottom_region', img_game_right_bottom_region)
            hue_dict, _ , _ = cv2.split(cv2.cvtColor(img_game_right_bottom_region, cv2.COLOR_BGR2HSV))
            mean_hue = np.mean(hue_dict)
            cv2.rectangle(img_game, (width_img_game - 20 , height_img_game - 20), (width_img_game, height_img_game ), (255, 255, 0), 4) # height_img_game - 20, width_img_game  (height_img_game, width_img_game

            # Получаем значения с ползунков
            try:
                lh = cv2.getTrackbarPos("L - H", "Trackbars")  # Low Hue
                ls = cv2.getTrackbarPos("L - S", "Trackbars")  # Low Saturation
                lv = cv2.getTrackbarPos("L - V", "Trackbars")  # Low Value
                hh = cv2.getTrackbarPos("H - H", "Trackbars")  # High Hue
                hs = cv2.getTrackbarPos("H - S", "Trackbars")  # High Saturation
                hv = cv2.getTrackbarPos("H - V", "Trackbars")  # High Value
            except cv2.error as e:
                print(f"Ошибка работы с ползунками: {e}")
                break

            # Создаем нижний и верхний пороги для HSV
            lower_bound = np.array([lh, ls, lv])
            upper_bound = np.array([hh, hs, hv])

            # Применяем маску
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            # Определяем ядро (структурирующий элемент)
            kernel = np.ones((10, 10), np.uint8)  # Размер ядра = 5x5

            # Применяем dilation
            mask = cv2.dilate(mask, kernel, iterations=1)
            # Находим связные компоненты
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Отображаем связные компоненты на изображении
            output = img.copy()
            local_object_coord = []
            if num_labels == 0:
                press_gap()
            for i in range(1, num_labels):
                x_comp, y_comp, w_comp, h_comp, area = stats[i]
                if area > 5:  # Минимальная площадь для фильтрации шума
                    cv2.rectangle(output, (x_comp, y_comp), (x_comp + w_comp, y_comp + h_comp), (0, 255, 0), 2)
                    cv2.putText(output, f"ID: {i} {area}", (x_comp, y_comp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    x_roi_2 = int(x_comp + w_comp // 2)
                    y_roi_2 = int(y_comp + h_comp // 2)
                    x_roi = x_roi_2 + roi_2[0]
                    y_roi = y_roi_2 + roi_2[1]
                    x_global = x_roi + roi[0]
                    y_global = y_roi + roi[1]

                    if flag_allow_click_mouse:
                        '''if i in dict_number_last_x_coord.keys():
                            x_global += x_global - dict_number_last_x_coord[i]
                        dict_number_last_x_coord[i] = x_global'''
                        delta_x = 0
                        if i in dict_number_last_x_coord.keys():
                            delta_x = x_global - dict_number_last_x_coord[i]
                            if delta_x < 0:
                                delta_x = 5
                            # print(dict_number_last_x_coord[i], x_global, x_global - dict_number_last_x_coord[i])

                        # x_global_click , y_global_click = x_global + delta_x, y_global
                        mouse_click(x_global + delta_x, y_global)
                        print("Клик мышкой")
                        # flag_allow_click_mouse_thread = True

                        number_of_shots -= 1
                        press_gap()
                        dict_number_last_x_coord[i] = x_global
                    cv2.circle(img_game, (x_roi, y_roi), 5, (255, 0, 0), 3)
                    local_object_coord.append([int(x_comp + w_comp // 2), int(y_comp + h_comp // 2)])

            # print(local_object_coord)

            # Отображаем результаты

            cv2.putText(img_game, f"{mean_hue}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
            cv2.imshow("Original ROI", img)
            cv2.imshow("Mask", mask)
            cv2.imshow("Detected Objects", output)
            cv2.imshow("img_game", img_game)

            # Выход по нажатию клавиши 'q'
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            if key == ord('r'):
                roi_2 = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
                print('roi_2', roi_2)

        # Очищаем ресурсы
        cv2.destroyAllWindows()


# Запуск функции
if __name__ == "__main__":
    # Запуск прослушивания клавиш через pynput
    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()

    capture_and_filter_roi()
