# Проект технического зрения по распознаванию объектов в полете
___
Программы представляет собой совместную работу библиотеки opencv с помощью которой происходит распознавание белого полотна на земле и сторонней программы tesseract от Google которая является обученной моделью распознавания текста с изображения с поддержкой множества языков.

![Структура ПО](https://github.com/PytAZU/Zrenie/blob/f7d17ccea2aada46bc9138d28e7421720fc82fac/%D0%A1%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%B0.jpg "Структура")

Во время полета программа каждую секунду делает последовательно 7 скриншотов которые поочередно обрабатываются функциями поиска полотна в кадре, преобработки обнаруженного квадрата для передачи его в функцию распознаванию символа на полотне и вычисления смещения полотна от центра аппарата для получения точного положения полотна.

Каждое изображение сразу после того как было сделано обрабатывается внутренними функциями ПО, если квадрат(ы) не был(и) обнаружен(ы), то изображения отбрасывается и происходит ожидание следующего, если а кадре найден(ы) квадрат(ы), то каждый из них отдельно вырезается с кадра, с помощью математических преобразований разворачивается в анфас и передается программе tesseract, которая возвращает обнаруженный на квадрате символ и процент своей "уверенности", эти данные записываются в отдельный файл вместе с данными с GPS трекера, поток данных с которых поступают в тот же файл вместе с обнаруженным квадратом. Параллельно с обработкой и подготовкой изображения для проекта нейросети отдельная функция вычисляет смещение квадрата/квадратов в кадре и исходя из данных о высоте полета вычисляет реальное смещение обнаруженного объекта относительно аппарата, прибавляя их к данным с GPS трекера для получения более точных значений координат обнаруженных символов.
___
## Установка
1. Для использования ПО необходимо заранее установить программу [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
3. В терминале перейдите в нужную (путую) директорию
```
cd path_to_your_directory
```
3. Скопируйте репозиторий в выбранную директорию
```
git clone https://github.com/PytAZU/Zrenie.git
```
