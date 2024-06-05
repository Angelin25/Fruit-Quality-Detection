import cv2
import tensorflow as tf

import os

DATADIR = "E:\data"
CATAGORIES = ["freshbanana","rottenbanana"] 

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array)
    plt.show()
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array)
    plt.show()

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("E:\data")

import os
cwd = os.getcwd()
path = os.path.join(cwd,DATADIR,CATAGORIES[0])

cwd = os.getcwd()

from tqdm import tqdm
import matplotlib.pyplot as plt
count = 1
path = os.path.join(DATADIR,CATAGORIES[0])
for img in tqdm(os.listdir(path)):   

  prediction = model.predict([prepare('E:\data\rottenbanana\rotated_by_15_Screen Shot 2018-06-12 at 8.47.57 PM.png')])
  print(prediction) 
  if int(prediction[0][0]) == 1:
    print("rotten")
  else:
    print("fresh")
  
  break

testing_data = []
IMG_SIZE = 50

CATEGORIES = ["freshbanana","rottenbanana"]
def create_testing_data():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category) 

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                testing_data.append([new_array, class_num])
            except Exception as e:
                pass
           
create_testing_data()

print(len(testing_data))

import random

random.shuffle(testing_data)
X = []
y = []
import numpy as np
for features,label in testing_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
import pickle

pickle_out = open("ATXNS.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("atyns.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

import pickle
pickle_in = open("ATXNS.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("atyns.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

results = model.evaluate(X, y, batch_size=128)
print('test loss, test acc:', results)

IMG_SIZE = 50

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
results = model.evaluate(X, y, batch_size=128)
print('test loss, test acc:', results)

#loss_train = history.history['train_loss']
#loss_val = history.history['val_loss']
epochs = range(1,35)
#plt.plot(epochs, loss_train, 'g', label='Training loss')
#plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from tqdm import tqdm
import matplotlib.pyplot as plt
count = 1
pred = []

for category in CATEGORIES:
  path = os.path.join(DATADIR,category)  
  print(path)
  for img in tqdm(os.listdir(path)):   
 
    if int(prediction[0][0]) == 1:
      pred.append("Rotten Banana")
    else:
      pred.append("Fresh Banana")

print(pred[:10])

import random

X = []
y = []
actual = []
import numpy as np
for features,label in testing_data:
    X.append(features)
    y.append(label)
print(y[900:])
for i in y:
  if i == 1:
      actual.append("Rotten Banana")
  else:
      actual.append("Fresh Banana")
print(actual[:10])

import csv
import pandas as pd

fruits = {'Actual Fruit': actual,
        'Predicted Fruit': pred
        }

df = pd.DataFrame(fruits, columns= ['Actual Fruit', 'Predicted Fruit'])

df.head()

from sklearn.utils import shuffle
df = shuffle(df)

df.head()
