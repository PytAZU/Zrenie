import cv2
import numpy as np
import math

def find_squares(image):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Используем метод Canny для обнаружения границ
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Находим контуры
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        # Аппроксимируем контур
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Если контур имеет 4 вершины и замкнут, считаем его квадратом
        if len(approx) == 4 and cv2.isContourConvex(approx):
            squares.append(approx)
    
    return squares

def order_points(pts):
    # Инициализируем список координат упорядоченных точек
    rect = np.zeros((4, 2), dtype="float32")
    
    # Верхняя левая точка будет иметь наименьшую сумму, а
    # нижняя правая точка будет иметь наибольшую сумму
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Теперь вычислим разницу между точками
    # Верхняя правая точка будет иметь минимальную разницу,
    # нижняя левая будет иметь максимальную разницу
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(frame, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Вычисляем ширину нового изображения
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Вычисляем высоту нового изображения
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Создаем массив точек назначения
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Вычисляем матрицу перспективного преобразования и применяем её
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    
    return warped


def four_point_transform(image, pts):
    # Получаем упорядоченный набор точек
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Вычисляем ширину нового изображения
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Вычисляем высоту нового изображения
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Теперь, когда у нас есть размеры нового изображения, создаем набор точек назначения
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Вычисляем матрицу перспективного преобразования и применяем её
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def is_contour_white(image, contour, threshold=200):
    # Создаем маску для контура
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Используем маску для получения пикселей внутри контура
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Проверяем, есть ли белые пиксели в маскированной области
    return np.any(masked_gray >= threshold)