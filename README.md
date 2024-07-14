# ***Проект технического зрения по распознаванию объектов в полете***

## ***Описание***

Программы представляет собой совместную работу библиотеки opencv с помощью которой происходит распознавание белого полотна на земле и сторонней программы tesseract от Google которая является обученной моделью распознавания текста с изображения с поддержкой множества языков.

![Структура ПО](https://github.com/PytAZU/Zrenie/blob/ad7a493039b995dfdbcd24d02ba3130c336c3ba4/%D0%91%D0%BB%D0%BE%D0%BA-%D1%81%D1%85%D0%B5%D0%BC%D0%B0.jpg "Структура")

Во время полета беспилотного летательного аппарата программа каждую секунду последовательно создает 7 высококачественных скриншотов. Эти изображения затем подвергаются многоуровневой обработке с использованием современных алгоритмов компьютерного зрения и анализа изображений. Первым этапом обработки является поиск полотна в каждом кадре, что достигается с помощью алгоритмов распознавания объектов, основанных на открытой библиотеке OPENCV.

Когда полотно успешно идентифицировано, начинается процесс предварительной обработки обнаруженного квадрата. Эта обработка включает в себя коррекцию перспективы и устранение геометрических искажений, что позволяет получить более четкое изображение для последующего анализа. Подготовленное изображение затем передается в OCR (Optical Character Recognition) систему Tesseract, которая с высокой точностью распознает символы на полотне. Tesseract возвращает распознанный символ вместе с процентом уверенности в правильности распознавания.

Все данные, полученные на этом этапе, включая координаты квадрата и распознанные символы, записываются в структурированный файл. Этот файл также содержит данные с GPS-трекера, который непрерывно фиксирует координаты аппарата. Поток данных с GPS-трекера синхронизируется с временными метками каждого кадра, что обеспечивает точное сопоставление визуальных данных и географических координат.

Параллельно с этим процессом, отдельная функция выполняет вычисление смещения квадрата(ов) относительно центра кадра. Эти вычисления включают анализ геометрических параметров изображения и использование данных о высоте полета для определения реального смещения обнаруженного объекта относительно аппарата. Данные о смещении квадрата(ов) интегрируются с информацией GPS, что позволяет корректировать координаты и получать максимально точные значения местоположения обнаруженных символов.

Одним из недостатков Tesseract является его неспособность распознавать символы, которые сильно повернуты на изображении. Чтобы обойти эту проблему, мы применяем метод многократного разворота. Сначала распознаем квадрат и устанавливаем его фронтально по отношению к камере. Затем, поворачиваем символ на 90 градусов четыре раза и каждый раз выполняем распознавание с помощью Tesseract. После этого сравниваем уровень уверенности для каждого из четырех распознаваний и выбираем наиболее точный результат, принимая его за окончательный.

Также возможна ситуация, когда камера фиксирует один и тот же символ с разных точек пространства. Чтобы избежать избыточности в итоговом файле с данными, применяется метод среднего положения объекта. Этот метод включает определение позиции одного и того же объекта на различных кадрах и усреднение полученных координат для получения наиболее точного положения объекта.
## ***Установка***

:exclamation:**Предупреждение**:exclamation:

При использовании программного комплекса необходимо в главном файле указать путь до установленной программы Tesseract на вашей машине

1. Для использования ПО необходимо заранее установить программу [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
3. В терминале перейдите в нужную (путую) директорию
```
cd path_to_your_directory
```
3. Скопируйте репозиторий в выбранную директорию
```
git clone https://github.com/PytAZU/Zrenie.git
```
## ***Использование***

**Windows**

1. Откройте исполняемый файл и раскомментируйте строку с указанием пути к Tesseract и укажите путь к нему на вашей машине
2. Откройте командную строку
3. Перейдите в директорию проекта
```
cd path_project_directory
```
3. Запустите основной файл
```
python main.py
```
