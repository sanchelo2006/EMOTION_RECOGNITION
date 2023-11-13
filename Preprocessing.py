import cv2
import numpy as np
import random
import os
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import fnmatch

#переменная - словарь классов
label_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise', 8: 'uncertain'}

#этот класс делает аугментацию
class AugImage:
    
    def __init__(self):
        
        pass
       
        
    def rotate_image(self, image_file, deg):
        
        image = cv2.imread(image_file)
        
        rows, cols,c = image.shape
    
        M = cv2.getRotationMatrix2D((cols / 2,rows / 2), deg, 1)
    
        image = cv2.warpAffine(image, M, (cols, rows))
    
        return image
    
    def erosion_image(self, image_file, shift):
        """
        значения от 3 до 6 дают нормальный результат
        """
        
        image = cv2.imread(image_file)
        
        kernel = np.ones((shift,shift),np.uint8)
    
        image = cv2.erode(image,kernel,iterations = 1)
    
        return image
    
    def gausian_blur(self, image_file, blur):
        
        image = cv2.imread(image_file)
        
        image = cv2.GaussianBlur(image,(5,5),blur)
    
        return image

    def dilation_image(self, image_file, shift):
        
        """
        значения от 3 до 5 дают приемлемый результат
        """
        
        image = cv2.imread(image_file)
        
        kernel = np.ones((shift, shift), np.uint8)
    
        image = cv2.dilate(image,kernel,iterations = 1)
    
        return image
    
    def random_func(self, image_file):
        
        """
        функция случайного выбора между другими функциями
        """
        func = random.choice([self.rotate_image(image_file, 7),
                               self.erosion_image(image_file, 5),
                               self.gausian_blur(image_file, 10),
                               self.dilation_image(image_file, 5)])
        
        #возвращает уже готовый numpy массив обработанного изображения!
        return func
    
#этот класс
#1 - вырезает лица из изображений
#2 - делает балансировку датасета
#3 - вычисляет медианные значения размеров изображений
#4 - препроцессинг изображения перед подачей в модель (при детекции). Решил, что эта функция более уместна здесь
class PreprocessDataset:
    
    def __init__(self):
        pass



    def crop_face(self, initial_path, final_path, user_confidence = 0.5):

        """
        назначение этой функции - вырезать лицо из изображения для более успешной тренировки сети

        Эта функция принимает на вход два пути: initial_path - путь где лежат папки с изображениями
        (названия папок - классы детекции), final_path - это название папки, где будут созданы папки
        с такими же названиями классов и в них будут сохраннены уже обработанные изображения


        dnn из библиотеки моделей opencv - используется для поиска лица на изображении
        в final_path - сохраняется обрезанное изображение лица

        user_confidence - задаем порог вероятности определения (рекомендуемый - 0.5)

        файлы: deploy.prototxt и res10_300x300_ssd_iter_140000.caffemodel необходимы для загрузки
        модели dnn. Они должны лежать в папке с этим кодом!
        """
        #загружаем dnn модель из зоопарка opencv
        net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './res10_300x300_ssd_iter_140000.caffemodel')

        #счетчики количества папок с классами
        counter = 0

        #проверяем есть ли такой путь в корневой папке
        if os.path.exists(initial_path):

            #пробегаем по папкам с классами
            for root, dirs, files in os.walk(initial_path):

                #сбрасываем счетчик файлов в папке
                counter_files = 0

                #первая папка - корневая папка
                if root == initial_path:

                    #берем из неё названия папок с классами
                    list_folders = dirs
                    continue

                #пробегаем по файлам по очереди в каждой папке классов
                for name in files:


                    #создаем путь: класс/название файла
                    core_path = os.path.join(list_folders[counter], name)

                    #считываем изображение
                    image = cv2.imread(str(initial_path) + '/' + str(core_path))

                    #считываем размеры изображения для последующей конвертации координат bounding box
                    (h, w) = image.shape[:2]

                    #здесь происходит детекция лица
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()

                    #пробегаем по всем обнаруженным детекциям
                    for i in range(0, detections.shape[2]):

                        #если вероятность больше установленной пользователем, то вырезаем этот bounding box
                        confidence = detections[0, 0, i, 2]

                        if confidence > user_confidence:

                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                    image_crop = image[startY:endY, startX:endX]

                    #создаем папку с классом
                    if not os.path.exists('./' + str(final_path) + '/' + str(list_folders[counter])):
                        os.makedirs('./' + str(final_path) + '/' +  str(list_folders[counter]))

                    #сохраняем обработанное изображение
                    name_new_file = str(counter_files) + '_' + str(list_folders[counter]) + '.jpg'
                    cv2.imwrite('./' + str(final_path) + '/' + 
                                str(list_folders[counter]) + '/' +  str(name_new_file), image_crop)

                    #обновляем счетчик файлов
                    counter_files += 1

                #заполняем словарь с количеством файлов в папках
                files_in_folders[list_folders[counter]] = counter_files

                #обновляем счетчик папок с классами
                counter += 1

        #если такого пути к необработанным изображениям нет - выводим текст       
        else:
            print(initial_path, ' - такого пути нет!')

        #cv2.waitKey(0)
        
    
    def balance_dataset(self, main_path):
    
        """
        эта функция производит балансировку датасета. Находит папку с максимальным количество файлов и
        с помощью аугментации добавляет в другие папки файлы. Здесь используется рандомная аугментация
        из класса AugImg
        """

        emotions_dict = dict()

        #проходим по основной папке, считаем кол-во файлов, создаем словарь
        for root, dirs, files in os.walk(main_path):

            if root == main_path:
                continue

            count = len(fnmatch.filter(os.listdir(root), '*.*'))

            emotions_dict[root] = count

        #определяем экземпляр аугментации
        my_aug = AugImage()

        #находим папку с максимальным количеством файлов
        max_feature = max(emotions_dict, key = emotions_dict.get)

        #определяем максимальное количество файлов
        max_number = max(emotions_dict.values())

        print('максимальное количество: ', max_number)

        #удалим из словаря ключ с фичей с максимальным значением
        del emotions_dict[max_feature]

        #проходим по словарю
        for item_path in emotions_dict:

            #этот счетчик нужен для названия создаваемых файлов
            counter = 0

            #вычисляем разницу между количеством файлов в этой папке и максимальным
            number = max_number - emotions_dict[item_path]

            print('для', item, ' - ', number, 'изначально')

            #если количество файлов в папке меньше чем в 2.5 раза - рекомендация - добавить другие файлы
            if number / emotions_dict[item] > 2.5:
                print('слишком большая разница в количестве для', item_path)
                print('лучше найти больше других изображений для этого класса!')
                continue

            #если разница меньше чем количество файлов в папке - то добавляем кол-во файлов равное этой разнице
            if number < emotions_dict[item_path]:

                #случайно выбираем файлы из папки для аугментации
                filenames = random.sample(os.listdir(item_path), number)

                counter = 0

                #проходим по выбранным файлам
                for file in filenames:

                    file_path = os.path.join(item_path, file)

                    #аугментация
                    img = my_aug.random_func(file_path)

                    #сохраняем созданный файл в ту же папку
                    cv2.imwrite(item_path + '/' + '_aug_' + str(counter) + '.jpg', img)

                    counter += 1

                print('для', item, ' - ', number, 'добавлено')

            #если разница больше чем файлов в папке, то берем просто количество файлов в папке
            elif number > emotions_dict[item_path]:

                filenames = random.sample(os.listdir(item_path), emotions_dict[item_path])

                counter = 0

                for file in filenames:

                    file_path = os.path.join(item_path, file)

                    img = my_aug.random_func(file_path)

                    cv2.imwrite(item_path + '/'  + '_aug_' + str(counter) + '.jpg', img)

                    counter += 1

                print('для', item_path, ' - ', emotions_dict[item_path], 'добавлено')

        print('датасет сбалансирован, проверьте папку ', main_path)
    
    def count_image_sizes(self, main_path):
    
        """
        main_path - путь к папке где хранятся папки с изображениями
        эта функция считывает размеры всех изображений в том числе и в подпапках от main_path
        рисует две гистограммы: ширина и длина изображений
        """

        list_heights = []
        list_width = []
        for root, dirs, files in os.walk(main_path):

            for file in files:
                filepath = os.path.join(root, file)
                img = Image.open(filepath)

                list_heights.append(img.height)
                list_width.append(img.width)

        fig, axs = plt.subplots(figsize=(12,6), ncols = 1, nrows = 2)
        axs[0].hist(list_heights)
        axs[1].hist(list_width)
        axs[0].set_title('Высота')
        axs[1].set_title('Ширина')
        plt.show() 
        print('медиана высоты = ', np.median(list_heights))
        print('медиана ширины = ', np.median(list_width))

    def process_image(self, image):
        
        (h, w) = image.shape[:2]
        
        if h ==0 or w == 0:
            
            return None
        
        if image is not None:
        
            image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)


            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image