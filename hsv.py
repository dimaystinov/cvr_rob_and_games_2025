import cv2
import numpy as np

# Функция для создания ползунков
def create_trackbars():
    # Создаем окно с ползунками для настройки HSV-порогов
    cv2.namedWindow("Mask")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda x: None)  # Low Hue
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)  # Low Saturation
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, lambda x: None)  # Low Value
    cv2.createTrackbar("H - H", "Trackbars", 179, 179, lambda x: None)  # High Hue
    cv2.createTrackbar("H - S", "Trackbars", 255, 255, lambda x: None)  # High Saturation
    cv2.createTrackbar("H - V", "Trackbars", 255, 255, lambda x: None)  # High Value

# Основная функция для обработки видео
def apply_hsv_filter():
    # Захват видео с камеры
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Создаем окно с ползунками
    create_trackbars()

    while True:
        # Считываем кадр с камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем кадр в цветовое пространство HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Получаем значения с ползунков
        lh = cv2.getTrackbarPos("L - H", "Trackbars")  # Low Hue
        ls = cv2.getTrackbarPos("L - S", "Trackbars")  # Low Saturation
        lv = cv2.getTrackbarPos("L - V", "Trackbars")  # Low Value
        hh = cv2.getTrackbarPos("H - H", "Trackbars")  # High Hue
        hs = cv2.getTrackbarPos("H - S", "Trackbars")  # High Saturation
        hv = cv2.getTrackbarPos("H - V", "Trackbars")  # High Value

        # Создаем нижний и верхний пороги для HSV
        lower_bound = np.array([lh, ls, lv])
        upper_bound = np.array([hh, hs, hv])

        # Применяем маску
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Находим связные компоненты (объекты)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Отображаем связные компоненты на изображении
        output = frame.copy()
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 400:  # Минимальная площадь для фильтрации шума
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, f"ID: {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отображаем исходное изображение, маску и результат
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Objects", output)

        # Выход по нажатию клавиши 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

# Запуск функции
if __name__ == "__main__":
    apply_hsv_filter()