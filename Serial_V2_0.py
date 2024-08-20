import cv2
import numpy as np
import os
import serial
import pytesseract
from time import sleep
import time
from concurrent.futures import ThreadPoolExecutor

# Устанавливаем язык для распознавания (армянский)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Укажите путь к Tesseract, если требуется
arm_lang = 'hye'  # Код для армянского языка

def get_gps_coordinates(gps_serial, timeout=5):
    """Пытается получить координаты GPS из последовательного порта в течение заданного времени."""
    latitude, longitude = None, None
    start_time = time.time()
    while time.time() - start_time < timeout:
        if gps_serial.in_waiting > 0:
            line = gps_serial.readline()
            if line:
                try:
                    line = line.decode('utf-8', errors='ignore')
                    latitude, longitude = parse_gpgga(line)
                    if latitude is not None and longitude is not None:
                        return latitude, longitude
                except UnicodeDecodeError:
                    pass  # Игнорируем ошибки декодирования
        sleep(0.1)  # Задержка в 0.1 секунды
    return latitude, longitude

def parse_gpgga(line):
    """Парсит строку GPGGA и возвращает широту и долготу."""
    if line.startswith('$GPGGA'):
        parts = line.split(',')
        # Проверяем наличие всех необходимых данных
        if len(parts) > 5 and parts[2] and parts[4]:
            try:
                latitude = parts[2]
                latitude_dir = parts[3]
                longitude = parts[4]
                longitude_dir = parts[5]
                
                # Преобразование координат в формат десятичных градусов
                lat_deg = float(latitude[:2])
                lat_min = float(latitude[2:])
                latitude = lat_deg + lat_min / 60.0
                if latitude_dir == 'S':
                    latitude = -latitude
                
                lon_deg = float(longitude[:3])
                lon_min = float(longitude[3:])
                longitude = lon_deg + lon_min / 60.0
                if longitude_dir == 'W':
                    longitude = -longitude
                
                return latitude, longitude
            except ValueError:
                print("Ошибка преобразования данных")
                return None, None
    return None, None

def detect_white_shapes(frame, lower_white=(0, 0, 200), upper_white=(180, 25, 255), contour_color=(0, 255, 0), contour_thickness=2):
    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создание маски для выделения белого цвета
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Поиск внешних контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисование только внешних контуров
    for contour in contours:
        # Аппроксимация контура для уменьшения количества точек
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Рисование контура на изображении
        cv2.drawContours(frame, [approx], 0, contour_color, contour_thickness)

    return frame, contours

def find_corner(frame, contours, center_lat, center_lon):
    """Находит углы контуров относительно центра изображения."""
    # Находим центр изображения
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Сохраняем координаты квадрата относительно центра изображения
    relative_coordinates = []

    for contour in contours:
        # Вычисляем моменты контура
        M = cv2.moments(contour)
        
        if M['m00'] != 0:
            # Координаты центра масс контура
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            
            # Относительные координаты центра контура относительно центра изображения
            relative_x = cX - center_x
            relative_y = center_y - cY  # Вверх - это вперед
            
            # Сохраняем относительные координаты
            relative_coordinates.append((relative_x, relative_y))
            
            # Отображаем координаты на изображении
            cv2.putText(frame, f"({relative_x},{relative_y})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return relative_coordinates

def display_image_with_gps(frame, latitude, longitude):
    """Отображает GPS координаты на изображении."""
    if latitude is None or longitude is None:
        text = "GPS: (None, None)"
    else:
        text = f"GPS: ({latitude}, {longitude})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def draw_crosshair_and_mark_contours(frame, contours, crosshair_color=(0, 0, 255), contour_color=(255, 0, 0)):
    """Рисует перекрестие в центре изображения и отображает контуры."""
    # Находим центр изображения
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Рисуем перекрестие в центре изображения
    cv2.line(frame, (center_x, 0), (center_x, height), crosshair_color, 1)  # Вертикальная линия
    cv2.line(frame, (0, center_y), (width, center_y), crosshair_color, 1)   # Горизонтальная линия

    # Обрабатываем каждый контур
    for contour in contours:
        # Рисуем центр контура на изображении
        cv2.drawContours(frame, [contour], -1, contour_color, 2)

    return frame

def recognize_armenian_text(image_path):
    """Распознает текст на армянском языке на изображении."""
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang=arm_lang)
    return text

def process_and_recognize_texts(squares_folder, output_file):
    """Проходит по сохраненным изображениям в папке и распознает текст на армянском языке, записывая данные в файл."""
    for square_image in os.listdir(squares_folder):
        image_path = os.path.join(squares_folder, square_image)
        if os.path.isfile(image_path):
            text = recognize_armenian_text(image_path)
            print(f"Распознанный текст на изображении {square_image}: {text}")
            write_to_file(output_file, "", text)  # Пустая строка для GPS данных, добавим ее позже

def write_to_file(filename, gps_data, recognized_text):
    """Записывает данные GPS и распознанный текст в файл."""
    with open(filename, 'a') as file:
        file.write(f"{filename}; {gps_data}; Распознанный символ: {recognized_text}\n")

def main():
    # Настройка камеры
    cap = cv2.VideoCapture(0)

    # Настройка последовательного порта для GPS
    gps_serial = serial.Serial('COM8', baudrate=9600, timeout=1)

    # Папка для сохранения квадратов
    squares_folder = 'Squares'
    
    # Имя файла для записи данных
    timestamp = time.strftime("%d_%m_%y_%H_%M_%S")
    output_file = f"data_{timestamp}.txt"

    # Настройка для работы с потоками
    with ThreadPoolExecutor() as executor:
        while True:
            start_time = time.time()

            # Захват кадра
            ret, frame = cap.read()
            if not ret:
                print("Не удалось захватить кадр")
                break

            # Запуск параллельных задач
            future_gps = executor.submit(get_gps_coordinates, gps_serial, timeout=1)
            future_contours = executor.submit(detect_white_shapes, frame)

            # Ожидаем завершения задач
            gps_coordinates = future_gps.result()
            frame, contours = future_contours.result()

            # Рисуем перекрестие и отмечаем центры контуров
            frame = draw_crosshair_and_mark_contours(frame, contours)

            # Если GPS данные получены, находим координаты углов и отображаем их
            if gps_coordinates[0] is not None and gps_coordinates[1] is not None:
                future_corners = executor.submit(find_corner, frame, contours, gps_coordinates[0], gps_coordinates[1])
                relative_coords = future_corners.result()
                gps_data = f"GPS: ({gps_coordinates[0]}, {gps_coordinates[1]})"
            else:
                relative_coords = []
                gps_data = "GPS: (None, None)"

            # Отображаем GPS координаты на изображении
            frame = display_image_with_gps(frame, gps_coordinates[0], gps_coordinates[1])

            # Распознаем текст на сохраненных изображениях
            future_recognize_text = executor.submit(process_and_recognize_texts, squares_folder, output_file)
            future_recognize_text.result()  # Ожидание завершения распознавания текста

            # Записываем данные GPS и распознанные символы в файл
            write_to_file(output_file, gps_data, "Text from previous images")

            # Отображаем результат
            cv2.imshow('Detected White Shapes', frame)

            # Задержка для достижения 13 кадров в секунду
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / 13.0) - elapsed_time)
            time.sleep(sleep_time)

            # Нажмите 'q', чтобы выйти из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

    if __name__ == '__main__':
        main()
