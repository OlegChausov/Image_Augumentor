print('Do you have Intel CPU: put 1 if yes')
cpu=input()
if cpu=='1':
    from sklearnex import patch_sklearn
from color_convert import color
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import dataframe_image
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
patch_sklearn()

def rgbToColor(*args):
    r,g,b=args
    return (r << 16) + (g << 8) + b

def rgb_int2tuple(rgbint):
    return (rgbint // 256 // 256 % 256, rgbint // 256 % 256, rgbint % 256)

print('Put youre file name here with extension. Its gotta be in this folder!')
filename = input()

print('Put youre test size part in range 0 to 1')
test_size_input=float(input())


with Image.open(filename) as colourImg:
    width = colourImg.size[0]
    height = colourImg.size[1]
    new_im = Image.new(mode="RGB", size=(width, height))
    draw = ImageDraw.Draw(new_im)
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
    indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
    allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
    df = pd.DataFrame(allArray, columns=["y", "x", "red","green","blue"])
    
    my_array=df.to_numpy()
    
    for i in my_array:
        temp=i[2:]
        i[2]=rgbToColor(*temp)
        
    my_array1=my_array[:,0:3]
    
    # Input Data
    x = my_array[:,0:2]
    # Output Data
    y = my_array[:,2]

    print('Choose the Sk-Learn Regression Kernel: 1-Linear; 2-Polynominal; 3- RBF')
    kern = input()
    if kern == '1':
        kern = 'Linear'
    elif kern == '2':
        kern = 'poly'
    elif kern == '3':
        kern = 'RBF'
        
        
    print('Now wait')

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_size_input, random_state = 42)
    if kern == 'Linear':
        regressor = SVR(kernel='linear')
    elif kern == 'poly':
        regressor = SVR(kernel='poly', C=1e3, degree=2)
    elif kern == 'RBF':
        regressor = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    regressor.fit(xtrain, ytrain)
    y_pred = regressor.predict(xtest).astype(int)
    
    predicted_array=np.column_stack([xtest, y_pred])
    trained_array=np.column_stack([xtrain, ytrain])
    
    for i in predicted_array:
        draw.point((i[0], i[1]), rgb_int2tuple(i[2]))
    for i in trained_array:
        draw.point((i[0], i[1]), rgb_int2tuple(i[2]))
    

        
    new_im.save("result_"+kern+"_kernel_"+filename, "PNG")
    
    
        
    
        


