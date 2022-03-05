# Intelligent Placer

## Постановка задачи

Имеется фотография подмножества предметов фиксированного множества и многоугольник. Необходимо определить, можно ли расположить все предметы в многоугольнике.

### Требования к фотографии предметов

1. Предметы не должны пересекаться
2. Предметы должны иметь четко выраженные границы (между предметами должна существовать цепочка из соседних пикселей фона, разделяющая предметы на фото)
3. Камера направлена перпендикулярно к поверхности, на которой расположены предметы
4. Все предметы расположены лицевой поверхностью вверх (как на фото ниже)
6. Все предметы должны быть полностью видны на фото 
7. Предметы должны располагаться на белой поверхности
8. Один объект может присутствовать на фото один раз
9. Расстояние от камеры до поверхности должна быть от 25 до 35 сантиметров 

### Способ задания многоугольника
1. Многоугольник задается списком координат вершин
2. Соседние вершины многоугольника должны идти подряд в списке вершин (первая идет за последней)
3. Координаты вершин многоугольника соовтветствуют сантиметрам на естественной плоскости

## Сбор данных

### Расположение
- Фото с подмножествами предметов находятся в папке /data/items_sets
- Файлы с заданными вершинами многоугольников в папке /data/polygons
- Собранные примеры (вход, выход) в файле /data/samples.json

### Генерация файлов с вершинами

- На листе А4 точками отмечаются вершины многоугольника
- Лист сканируется на принтере с настройками соответствия листа и скана
- Отсканированное изображение бинаризуется, выделяются компоненты связности и находятся их центры
- Центры вручную сортируются в соответствии с требованиями

### Набор предметов
![](./images/items/box.jpg)
![](./images/items/card.jpg)
![](./images/items/coin.jpg)
![](./images/items/keychain.jpg)
![](./images/items/lighter.jpg)
![](./images/items/mediator.jpg)
![](./images/items/pen.jpg)
![](./images/items/phone.jpg)
![](./images/items/power_bank.jpg)
![](./images/items/record_book.jpg)
