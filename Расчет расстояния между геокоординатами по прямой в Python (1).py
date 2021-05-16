#!/usr/bin/env python
# coding: utf-8

# ## Расчет расстояния между геокоординатами по прямой в Python

# Считаем расстояния между объектами по их координатам. В данном примере будем считать расстояния от населенных пунктов до школ и выбирать ближайшую школу. 
# 
# Важно: расстояния считаются по прямой, без учета дорог, гор, рек и пр., поэтому дают только общее представление о дальности объектов. Чтобы точно рассчитать дистанцию и время в пути по дорогам, можно воспользоваться API матрицы расстояний Яндекс.Карт (платно, либо можно попробовать попросить бесплатно в исследовательских и научных целях по индивидуальному запросу) или OSRM — Open Sourced Routing Machine (бесплатно)

# ### 1. Скачивание данных

# 1. Данные по населенным пунктам скачиваем на сайте ИНИД (Инфраструктура научно-исследовательских данных) https://www.data-in.ru/data-catalog/datasets/160/
# 2. Данные по школам — парсингом сайта https://schoolotzyv.ru/schools/9-russia/ (данные доступны в формате json по ссылке https://drive.google.com/file/d/18PQTD4upgjS-zfcPgKWEj1_zWJz7RL64/view?usp=sharing) По некоторым регионам на нем есть не все школы. Можно дополнить данными с сайтов https://arhangelsk.fulledu.ru/, https://russiaedu.ru/, набора открытых данных с лицензиями Рособрнадзора (требуется парсинг xml) http://obrnadzor.gov.ru/otkrytoe-pravitelstvo/opendata/7701537808-fbdrl/ и других источников.

# ### 2. Загрузка библиотек

# In[2]:


import pandas as pd # будем работать в pandas
import numpy as np # numpy и sklearn для математических расчетов и использования формул
import sklearn.neighbors
import json # для работы с файлами в формате json


# ### 3. Загрузка и подготовка данных

# In[8]:


# загружаем датасет со школами в pandas

with open('/Users/me/Downloads/schoolotzyv.json', 'r') as read_file:
    otzyv = json.load(read_file)


# In[83]:


otzyv = pd.DataFrame(otzyv)
otzyv


# In[84]:


# нам нужно узнать регионы, для этого выбираем любую ссылку, переходим по ней и смотрим, как в ссылках зашит нужный нам регион

otzyv.loc[1,'url']


# In[85]:


# для примера возьмем Нижегородскую область (если взять всю Россию, считаться будет долго)
# она в ссылках на школы помечена как "146-nizhegorodskaya"
# выбираем только те строки, которые содержат такую запись
# подробнее о том как фильтровать датафреймы в pandas — в уроке https://istories.media/workshops/2021/03/05/python-biblioteka-pandas-chast-1/


schools_nn = otzyv[otzyv['url'].str.contains('146-nizhegorodskaya')]
len(schools_nn)


# In[65]:


schools_nn


# In[ ]:





# In[86]:


# поскольку мы выбрали только некоторые строки, индексы датафрейма сбились
# чтобы они опять начинались с нуля, нужно их сбросить методом reset_index

schools_nn = schools_nn.reset_index(drop = True)


# In[66]:


# загружаем датасет с населенными пунктами

places = pd.read_csv('/Users/me/Downloads/places.csv')
places


# In[78]:


# для примера выбираем только Нижегородскую область и только один район — например, Воскресенский 
# (чтобы точно всё быстро посчиталось на небольшом объеме данных)

places_voskresensk = places[(places['region'] == 'Нижегородская область') & 
                            (places['municipality'] == 'Воскресенский район')]
len(places_voskresensk)


# In[79]:


# опять обновляем индексы, чтобы шли с нуля 

places_voskresensk = places_voskresensk.reset_index(drop = True)


# ### 4. Создание словарей

# Прежде чем считать расстояния, нам понадобится создать два словаря — с населенными пунктами и со школами. Это нужно, чтобы в дальнейшем мы могли сопоставить разные датафреймы по какому-то уникальному параметру, доставать и добавлять нужные нам данные. Это нам пригодится на последних этапах. 
# 
# Для населенных пунктов таким уникальным параметром будет код ОКТМО (Общероссийского классификатора территорий муниципальных образований)
# 
# Перед этим нам нужно убедиться, что:
# 1) Во всех нужных нам столбцахуказан правильный тип данных
# 
# 2) Удалить дубликаты в тех столбцах, которые будут уникальным ключом для словаря (для населенных пунктов это поле oktmo, для школ — url, ссылка)

# In[80]:


# смотрим какой тип данный в столбце 'oktmo'

places_voskresensk['oktmo'].dtype


# In[81]:


# float, а нам нужны строки, поэтому меняем сначала на integer, а затем на str


# In[17]:


places_voskresensk['oktmo'] = places_voskresensk['oktmo'].astype('int').astype('str')
places_voskresensk['oktmo'].dtype


# In[73]:


# удаляем дубликты и сбрасываем индексы 

places_voskresensk[places_voskresensk['oktmo'].duplicated()]


# In[29]:


places_voskresensk = places_voskresensk[~places_voskresensk['oktmo'].duplicated()]
places_voskresensk = places_voskresensk.reset_index(drop = True)


# In[30]:


# собираем датафрейм с населенными пунктами в словарь 


places_by_oktmo = {} # пустой словарь 
for i in range(len(places_voskresensk)): # i это индекс, порядковый номер строки
    el = places_voskresensk.loc[i] # записываем в переменную el строку с индексом i 
    places_by_oktmo[el['oktmo']] = el # создаем словарь, где ключом будет код октмо, 
                                    # а значением — соответствующая ему строка датафрейма
len(places_voskresensk)


# In[ ]:


# (чтобы посмотреть, что содержится в el, попробуйте вывести на экран в отдельной ячейке places_voskresensk.loc[1] (или другое число))


# In[72]:


places_voskresensk.loc[1]


# In[31]:


places_by_oktmo['22622416166'] # пример того что выдает словарь по ключу - коду октмо


# In[ ]:


# переходим к датафрейму со школами 
# сначала меняем тип данных в столбцах с широтой и долгой — были строки, а нужны числа (float)


# In[87]:


schools_nn['geo_lat'].dtype


# In[33]:


schools_nn['geo_lat'] = schools_nn['geo_lat'].astype('float')
schools_nn['geo_long'] = schools_nn['geo_long'].astype('float')
print(schools_nn['geo_lat'].dtype)
print(schools_nn['geo_long'].dtype)


# In[26]:


# удаляем дубликаты 
schools_nn[schools_nn['url'].duplicated()]


# In[35]:


schools_nn = schools_nn[~schools_nn['url'].duplicated()]
schools_nn = schools_nn.reset_index(drop = True)


# In[ ]:


# собираем такой же словарь, как с населенными пунктами, только ключом будет url школы 


# In[36]:


schools_by_url = {}
for i in range(len(schools_nn)):
    el = schools_nn.loc[i]
    schools_by_url[el['url']] = el
len(schools_by_url)


# ### 5. Расчет матрицы расстояний

# Переходим к расчету расстояний между каждым населенным пунктом и каждой школой (матрица расстояний). За основу взят туториал Dana Lindquist https://medium.com/@danalindquist/finding-the-distance-between-two-lists-of-geographic-coordinates-9ace7e43bb2f

# In[37]:


# переводим координаты из градусов в радианы, потому что математические формулы обычно требуют 
# значения в радианах, а не в градусах.
# делаем это с помощью библиотеки numpy

schools_nn[['lat_radians_A','long_radians_A']] = (
    np.radians(schools_nn.loc[:,['geo_lat','geo_long']])
)
places_voskresensk[['lat_radians_B','long_radians_B']] = (
    np.radians(places_voskresensk.loc[:,['latitude_dd','longitude_dd']])
)


# In[38]:


# считаем расстояния с помощью формулы из библиотеки scikit-learn
# она вычисляет гаверсинусное расстояние, то есть представляет форму Земли как идеальную сферу (а не геоид, 
# как на самом деле), и за счет этого обеспечивает быстрые вычисления
# если требуется измерение в км, то нужно умножить на 6371, если в милях — на 3959


dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
dist_matrix = (dist.pairwise
    (schools_nn[['lat_radians_A','long_radians_A']],
     places_voskresensk[['lat_radians_B','long_radians_B']])*6371
)


# In[40]:


# создаем матрицу расстояний — таблицу, в которой индексами будут url школ,
# колонками - октмо населенных пунктов, а в ячейках будет расстояние от каждого населенного пункта до каждой школы

df_dist_matrix = (
    pd.DataFrame(dist_matrix,index=schools_nn['url'], 
                 columns=places_voskresensk['oktmo'])
)


# In[41]:


df_dist_matrix


# In[42]:


# нам нужно расстояние от населенных пунктов до школ, а не наоборот, поэтому транспонируем таблицу —
# меняем индексы и колонки местами


# In[43]:




df_dist_matrix = df_dist_matrix.T
df_dist_matrix


# ### 6. Выбор ближайшего объекта

# Выбираем ближайшую школу к каждому населенному пункту.

# In[44]:


# дальше мы сравниваем каждое значение в строке с предыдущим, чтобы проверить, больше оно или меньше. Меньшее записываем в переменную

schools_and_min_value_by_oktmo = {} # пустой словарь, где ключом будет код октмо, 
                                    #значениями — минимальное расстояние

columns = df_dist_matrix.columns # список с названиями колонок (url школ)
for current_oktmo in df_dist_matrix.index:
    distances = df_dist_matrix.loc[current_oktmo]
    min_distance = None # нужно для начала отсчета, чтобы было с чем сравнивать первое значение в строке
    min_j = None
    for j in range(len(columns)):
        if min_distance is None or min_distance > distances[j]: # сравниваем каждое значение в строке с 
            # предыдущим, чтобы проверить, больше оно или меньше. Меньшее записываем в переменную
            min_distance = distances[j]
            min_j = j 
    schools_and_min_value_by_oktmo[current_oktmo] = (columns[min_j], min_distance)
# заполняем словарь: ключ — код октмо, значение — url ближайшей школы и расстояние до нее
len(schools_and_min_value_by_oktmo)


# In[45]:


schools_and_min_value_by_oktmo['22622440151'] # пример того, что выдает словарь по коду октмо


# In[50]:


# добавляем колонки с мин расстоянием и url школы в датафрейм places_voskresensk (с населенными пунктами)
# создаем списки с полученными выше url и км, чтобы затем добавить их к датафрейму с населенными пунктами

school_url_column = [] 
min_distance_column = []
for oktmo in places_voskresensk['oktmo']:
    (url, distns) = schools_and_min_value_by_oktmo[oktmo]
    school_url_column.append(url) # берем из созданного выше словаря url ближайшей школы(url) и км до нее (distns),
    # добавляем их в словари в том порядке, в каком соответствующие им коды октмо расположены в датафрейме places_voskresensk
    min_distance_column.append(distns)
places_voskresensk['school_url_column'] = school_url_column # добавляем новую колонку с url ближайших школ
places_voskresensk['min_distance_column'] = min_distance_column # добавляем новую колонку с км до этих школ
len(places_voskresensk)


# In[46]:


places_voskresensk


# In[52]:


# из словаря со школами достаем названия школ по url и тоже добавляем их к датафрейму places_voskresensk новой колонкой


school_name = []

for url in places_voskresensk['school_url_column']:
    school_name.append(schools_by_url[url]['text'])
places_voskresensk['school_name'] = school_name


# In[54]:


# в конце появились 3 новые колонки — url ближайшей школы, сколько до нее км, название школы

places_voskresensk


# In[55]:


# сохраним полученный результат в csv 


places_voskresensk.to_csv('voskresensk_closest_schools.csv')


# ### 7. Анализ данных

# In[56]:


# Дальше можно анализировать данные.
# Например, выбрать 15 наиболе удаленных от школ объектов методов nlarest
# подробнее об nlargest и nsmallest https://istories.media/workshops/2021/03/19/python-biblioteka-pandas-chast-3/ 


# In[57]:


places_voskresensk.nlargest(15,'min_distance_column')


# In[60]:


# или посчитать, сколько детей в Воскресенском районе Нижегородской обл. живут дальше 5 км от школы

km5 = places_voskresensk[places_voskresensk['min_distance_column'] > 5]
print(places_voskresensk['children'].sum()) # всего детей
print(km5['children'].sum()) # живут дальше 5 км


# In[61]:


# или среднее расстояние от населенного пункта до школы 

places_voskresensk['min_distance_column'].mean()


# Готово, спасибо, что дошли до конца!
