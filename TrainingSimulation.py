import matplotlib.pyplot as plt
print('setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
from sklearn.model_selection import train_test_split


#step1
path = 'MyData'
data = importDataInfo(path)

#step2
data = balanceData(data, display=False)

#step3
imagesPath, steerings = loadData(path, data)
#print(imagesPath[0], steering[0])

#step4
xtrain, xval, ytrain, yval = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xtrain))
print('Total Validation Images: ', len(xval))

#step5


#step6

#step7

#step8
model = creatModel()
model.summary()

#step9
history = model.fit(batchGen(xtrain, ytrain, 100, 1), steps_per_epoch=300, epochs=10,
          validation_data=batchGen(xval, yval, 100, 0), validation_steps=200)

#step10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

