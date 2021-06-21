from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
import numpy as np
from tensorflow import Graph

img_height, img_width = 150,150
with open('./models/classes.json','r') as f:
    labelInfo=f.read()
labelInfo = json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/resnet101v2.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def checkModel(request):
    print (request)
    print (request.POST.dict())
    fileObj = request.FILES['filePath']
    type_class = request.POST['type_class']
    comb_class = request.POST['comb_class']

    if type_class == 1:
        if comb_class == 1 or comb_class 2 or comb_class == 3:
            error = 'Invalid input!'
        else:
            predictImage(request, fileObj)
    elif type_class == 2:
        if comb_class == 1:
            model2 = load_model('./models/resnet101.h5')
        elif comb_class == 2:
            model3 = load_model('./models/efficientnetl2.h5')
        elif comb_class == 3:
            model4 = load_model('./models/vgg16.h5')
    
def predictImage(request, fileObj):
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)
    a = np.max(predi[0][0])
    b = np.max(predi[0][1])
    c = np.max(predi[0][2])
    s_a = '{:.5f}'.format(100*a)
    s_b = '{:.5f}'.format(100*b)
    s_c = '{:.5f}'.format(100*c)
    predictedLabel = labelInfo[str(np.argmax(predi[0]))]
    score = np.max(predi[0])
    predictedScore = '{:.5f}'.format(100*score)

    context = {'filePathName':filePathName,
        'predictedLabel':predictedLabel[1],
        'predictedScore':predictedScore,
        'stats_a':s_a,
        'stats_b':s_b,
        'stats_c':s_c
    }
    return render(request,'predict.html',context) 

def viewDataBase(request):
    import os
    listOfImages = os.listdir('./media/')
    listOfImagesPath =['./media/'+i for i in listOfImages]
    context = {'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html', context) 