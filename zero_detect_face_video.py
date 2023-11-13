# импорт необходимых библиотек
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import Preprocessing as pp
import tensorflow as tf
from collections import deque

#берем из файла Preprocessing переменную названий классов и создаем экземпляр одного из классов из этого файла
labels_dict = pp.label_dict
preprocess = pp.PreprocessDataset()

#флаг для условий добавления предсказаний в словарь
flag_add = False

#функция делающая предсказание на основе модели
def predictions(image, labels_dict, model, flag, pred = []):
    
    """
    image - изображение на входе должно быть уже в формате массива!
    labels_dict - словарь классов (ключ - значение предсказания, value - название класса)
    model - модель нейронной сети делающая предсказание
    flag - для определения действия (или мы усредняем уже предсказания или просто добавляем новое)
    pred - для того же, что и флаг
    """
    image = preprocess.process_image(image)
    
    if image is None:
        
        return None
    
    #сначала готовим изображение к подаче в модель
    my_image = image.reshape((1, image.shape[0], image.shape[1]))
    

    #делаем предсказание
    #prediction - список со всеми вероятностями (9 вероятностей)
    if flag == 'pred':
        prediction = model.predict(my_image)
        return prediction
    
    elif flag =='answer':
        prediction = pred

        #конвертируем предсказание в название класса
        predict_convert = (prediction > 0.5).astype('int32')

        if np.sum(predict_convert) == 0:
            return None

        answer = labels_dict[np.where(predict_convert==1)[1][0]]

        return answer


# создаем необходимые аргументы командной строки
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--numberpixels', default = 60,
                help = 'количество пикселей для сравнения кадров (лицо смещается в кадре - но это все тоже лицо)')
ap.add_argument('-f', '--frames', default = 3,
                help = 'количество кадров для усреднения предсказания')
ap.add_argument("-p", "--prototxt", required = True,
	help = "путь до файла Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True,
	help = "путь до файла Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5,
	help = "минимальный порог вероятности для определения лица в кадре")
ap.add_argument("-v", "--camera", type = int, default = 0,
    help = "номер камеры для cv2")
ap.add_argument('-l', '--maxfaces', default = 5,
                help = 'макс количество лиц в кадре')
args = vars(ap.parse_args())

# грузим модели
print("[ИНФО] грузим модели...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
model = tf.keras.models.load_model('./model_emotions')

# включаем камеру и открываем окно для показа
print("[ИНФО] запускаем камеру...")
cap = cv2.VideoCapture(args["camera"])
time.sleep(2.0)
cv2.namedWindow('EMOTIONS')

#параметры контраста и освещения кадра
cv2.createTrackbar('brightness', 'EMOTIONS', 255, 510, (lambda a: None))
cv2.createTrackbar('contrast', 'EMOTIONS', 0, 100, (lambda a: None))

#список цветов для openCV. Нужен для случайного выбора цвета
colors = [(22, 158, 24), (155, 97, 26), (30, 138, 168), (8, 40, 137), (237, 186, 35)]

# очередь предсказаний эмоций лиц
# состоит из кортежей. Каждый кортеж состоит из:
# 1 - очередь из предсказаний одного и того же лица для усреднения
# 2 - координаты центра лица (для определения уникальности лица в кадре)
main_queue = deque(maxlen = args['maxfaces'])


# loop over the frames from the video stream
while True:
    
    # флаг для обозначения того, что мы добавили предсказание того же лица в очередь
    flag_add = False
    
    key = cv2.waitKey(1) & 0xFF
    
    # захватываем кадр с камеры и ресайзим его до 800 пикселей (можно тоже сделать настраиваемым параметром)
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 800)
    
    #настраиваем с помощью трекбаров яркость и контраст
    beta = cv2.getTrackbarPos('contrast', 'EMOTIONS')
    alpha = cv2.getTrackbarPos('brightness', 'EMOTIONS')
    alpha = alpha / 255
    beta = beta - 50
    
    frame = cv2.convertScaleAbs(frame, alpha = alpha, beta = beta)
    
    
    # берем размеры кадра и ковертим кадр в blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # прогоняем blob через сеть и находим лица
    net.setInput(blob)
    detections = net.forward()
    
# проходим в цикле по детекциям в кадре
    for i in range(0, detections.shape[2]):
        
        # берем вероятность предсказания детекции
        confidence = detections[0, 0, i, 2]
        
        # отфильтровываем только детекции с вероятностью выше установленной
        if confidence < args["confidence"]:
            continue
            
        # берем координаты bounding box детекции
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
       
        #определяем центр box`а
        centerX = int((endX - startX) / 2 + startX)
        centerY = int((endY - startY) / 2 + startY)
            
        
        # проверяем не пустая ли у нас очередь
        if len(main_queue) == 0:
            
            # если пустая - добавляем элемент
            main_queue.append((deque(maxlen = args['frames']), centerX, centerY))
            
        else:
            
            #иначе пробегаем по элементам очереди и определяем если есть тоже лицо что было в пред кадре
            for item in main_queue:
                
                if centerX < item[1] + args['numberpixels'] and\
                   centerX > item[1] - args['numberpixels'] and\
                   centerY < item[2] + args['numberpixels'] and\
                   centerY > item[2] - args['numberpixels']:
                    
                    # меняем флаг, чтобы обойти следующее условие (добавление нового лица)
                    flag_add = True
                    
                    # вырезаем лицо из кадра и делаем предсказание эмоции
                    image_crop = frame[startY:endY, startX:endX]
                    my_predictions = predictions(image_crop, labels_dict, model, 'pred')
                    
                    # добавляем предсказание в очередь (первый элемент нашей главной очереди из лиц)
                    item[0].append(my_predictions)
                    
                    # блок для усреднения предсказаний эмоций по нескольким кадрам---
                    temp_array = np.array(item[0])
                    avg_pred = np.max(temp_array, axis = 0)
                    my_predict = predictions(image_crop, labels_dict, model, 'answer', pred = avg_pred)
                    #----------------------------------------------------------------
                    
                    # блок для прорисовки рамки и названия эмоции---
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    (W, H), _ = cv2.getTextSize(my_predict, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame, (startX - 3, startY - 30),
                                  (startX + W + 10, startY), colors[i], -1)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[i], 5)
                    cv2.putText(frame, my_predict, (startX, y),cv2.FONT_HERSHEY_COMPLEX, 0.65, (252, 252, 252), 2)
                    #------------------------------------------------
                    
            # если очередь не пустая и такого элемента нет - то добавляем новый
            if flag_add == False:
                
                main_queue.append((deque(maxlen = args['frames']), centerX, centerY))


    # в верхнем углу показываем подсказку (q - exit)
    cv2.rectangle(frame, (0, 0), (100, 25), (0, 255, 0), -1)
    cv2.putText(frame, 'q - exit', (2, 15), cv2.FONT_HERSHEY_COMPLEX, 0.65, (252, 252, 252), 2)
    
    # показываем выход видео
    cv2.imshow("EMOTIONS", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # выход по нажатию клавиши 'q'
    if key == ord("q"):
        
        break
        
# надо подчистить все окна
cv2.destroyAllWindows()
