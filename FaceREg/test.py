#make prediction for custom image out of test set
import tensorflow as tf

import keras
from keras.models import Sequential
#from keras.models import load
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1




img = image.load_img("abc.jpg", grayscale=True, target_size=(48, 48))
model_name='meramodel'

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255
model=load_model(model_name)
print('model loaded')
custom = model.predict(x)
print(custom)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

    



plt.gray()


# In[ ]:


#plt.imshow(x)


# In[ ]:


plt.show()



# In[ ]:


max_val=custom.max()

print(max_val)

#custom_index = custom.any(max_val)

'''if custom[0] == max_val:
    print("angry")
elif custom[1] == max_val:
    print("disgust")
elif custom[2] == max_val:
    print("fear")
elif custom[3] == max_val:
    print("happy")
elif custom[4] == max_val:
    print("sad")
elif custom[5] == max_val:
    print("suprise")
elif custom[6] == max_val:
    print("neutral")
else:
    print("emotion")
        '''