import cv2
import numpy as np
import os
from square_rotating_test import perspective_transform
import gps

def get_coordinates():
    # Создаем GPS-сессию
    session = gps.gps(mode=gps.WATCH_ENABLE)
    
    try:
        # Получаем следующий набор данных, блокируем выполнение пока данные не будут получены
        report = session.next()
        
        # Проверяем, содержит ли отчет данные о местоположении
        if report['class'] == 'TPV':
            latitude = getattr(report, 'lat', None)
            longitude = getattr(report, 'lon', None)
            return latitude, longitude
    except StopIteration:
        # GPSD остановился, возможно, GPS приемник отключен
        return None, None

def detect_white_shapes(image_path, lower_white=(0, 0, 200), upper_white=(180, 25, 255), contour_color=(0, 255, 0), contour_thickness=2):
    # Чтение изображения
    image = cv2.imread(image_path)

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
        cv2.drawContours(image, [approx], 0, contour_color, contour_thickness)

    return image, contours, cv2.imread(image_path)  # Возвращаем также исходное изображение для вырезки контуров

def save_contours(contours, original_image, output_dir="Squares"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, contour in enumerate(contours):
        # Создание ограничивающего прямоугольника для каждого контура
        x, y, w, h = cv2.boundingRect(contour)
        # Вырезка области контура из исходного изображения
        contour_image = original_image[y:y+h, x:x+w]
        # Сохранение изображения контура
        contour_image_path = os.path.join(output_dir, f"contour_{i}.png")
        cv2.imwrite(contour_image_path, contour_image)

def find_corner(image, contours):
    # Находим центр изображения
    height, width, _ = image.shape
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
            relative_y = cY - center_y
            relative_coordinates.append((relative_x, relative_y))
            
            # Отображаем координаты на изображении
            cv2.putText(image, f"({relative_x},{relative_y})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return relative_coordinates

def display_image_with_gps(image, latitude, longitude):
    # Отображаем GPS координаты на изображении
    cv2.putText(image, f"GPS: ({latitude}, {longitude})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def save_coordinates_to_file(latitude, longitude, relative_coords, file_path="coordinates.txt"):
    with open(file_path, 'w') as file:
        file.write(f"GPS Coordinates: Latitude = {latitude}, Longitude = {longitude}\n")
        file.write("Relative Coordinates of Contours:\n")
        for i, (rel_x, rel_y) in enumerate(relative_coords):
            file.write(f"Contour {i}: ({rel_x}, {rel_y})\n")

def main():
    image_path = 'C:\\Study\\square_rotating_test\\images\\sq_test.png'

    # Получаем GPS координаты
    latitude, longitude = get_coordinates()
    if latitude is None or longitude is None:
        print("Не удалось получить координаты GPS")
        return

    # Обнаружение белых контуров
    output_image, contours, original_image = detect_white_shapes(image_path)
    
    # Сохранение контуров в качестве отдельных изображений
    save_contours(contours, original_image)

    # Находим координаты
    relative_coords = find_corner(output_image, contours)

    # Сохранение координат в файл
    save_coordinates_to_file(latitude, longitude, relative_coords)

    # Отображаем GPS координаты на изображении
    output_image = display_image_with_gps(output_image, latitude, longitude)

    # Показ результата
    cv2.imshow('Detected White Shapes', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
