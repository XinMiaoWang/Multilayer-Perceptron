import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import globalVar
import TkGUI as gui
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

import copy
import math

from collections import Counter

# 讀檔
def readfile():
    data = []
    filename = globalVar.path_
    with open(filename) as file:
        line = file.readline()
        while line:
            eachline = line.split()
            read_data = [ float(x) for x in eachline[0:len(eachline)-1] ] # 轉float
            read_data.insert(0, -1) # 所有data前面都加上-1
            label = [ int(x) for x in eachline[-1] ] # label轉int
            read_data.append(label[0])
            data.append(read_data)
            line = file.readline()

    # print('Original: ',data)
    return data

def preprocess(data):
    data, noiseData = romoveNoise(np.array(data)) # 移除雜點
    data = changeLabel(np.array(data)) # 統一label
    np.random.shuffle(data) # 打亂data
    # print('Random: ',data)
    train, test = data[:int(len(data)*2/3)], data[int(len(data)*2/3):] # 分2/3 teaing data，1/3 testing data
    # train, test = np.array(data[:]), []
    # print('Train: ',train)
    # print('Test: ',test)

    if len(noiseData) == 0:
        return train, test, []

    return train, test, noiseData

# 移除雜點
def romoveNoise(data):
    originalData = copy.copy(data)
    countLabel = Counter(data[:, -1]) # 計算label種類
    # print('QQQQQQ', countLabel)
    # print('Len: ',len(countLabel))

    if len(countLabel) > 2:
        most_common_words = [word for word, word_count in Counter(countLabel).most_common()[:-2:-1]] # 找出數量最少的label
        # print(type(most_common_words)) # list
        float_lst = [int(float(x)) for x in most_common_words] # 要先轉float再轉int否則會報錯

        removeIdx = np.where(data[:, -1] == float_lst[0]) # 找出雜點index
        print('Idx: ',removeIdx)
        noiseData = originalData[removeIdx,:]
        # print('!!!!!!!!', noiseData)
        result = np.delete(data, removeIdx, 0) # 刪除雜點
        # print('/////////////////////////////')
        # print('After Remove: ',result)
        # print('/////////////////////////////')

        return result,noiseData

    return data, []


# 統一label，原始label大的標為1，小的標為0
def changeLabel(data):
    changedata = copy.copy(data)
    # countLabel = Counter(changedata[:,-1])
    # print('QQQQQQ',countLabel)
    # print(data[:,-1])
    bigLabel = np.max(data[:,-1])
    bigIdx = np.where(data[:, -1] == bigLabel)
    print('bigIdx : ',bigIdx)
    changedata[bigIdx,-1] = 1

    smallLabel = np.min(data[:, -1])
    smallIdx = np.where(data[:, -1] == smallLabel)
    print('smallIdx : ', smallIdx)
    changedata[smallIdx, -1] = 0

    return changedata


def rmse(prediction, target):
    difference = prediction - target
    difference_squared = difference ** 2
    mean_of_differences_squared = difference_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)

    return rmse_val

def sgn(value):
    if value>=0.5:
        return 1
    else:
        return 0


def multilayer_perceptron(train, test, learning_rate, iteration, progressbar):

    maxValue = iteration
    currentValue = 0
    progressbar["value"] = currentValue
    progressbar["maximum"] = maxValue - 1

    # learning_rate = 0.8
    # weight = np.array([-1, 0, 1])
    weight_1 = np.round(np.random.rand(train.shape[1] - 1), 2) #隨機產生在[0,1)之間均勻分布的鍵結值
    weight_2 = np.round(np.random.rand(train.shape[1] - 1), 2)
    weight_3 = np.round(np.random.rand(train.shape[1] - 1), 2)
    # print('Original Weight: ',weight_1)

    train_error_rate = 0
    test_error_rate = 0

    x_train = train[:, :train.shape[1]-1]
    y_train = train[:, -1]
    # print('X_Train: ', x_train)
    # print('Y_Train: ', y_train)

    x_test = test[:, :test.shape[1] - 1]
    y_test = test[:, -1]
    # print('X_Test: ', x_test)
    # print('Y_Test: ', y_test)

    for n in range(iteration):
        progressbar["value"] = n
        progressbar.update()

        i = n % train.shape[0]

        v1 = np.dot(weight_1, x_train[i])
        y1 = 1 / ( 1 + math.exp(-v1) )
        v2 = np.dot(weight_2, x_train[i])
        y2 = 1 / ( 1 + math.exp(-v2) )

        gamma = np.dot( weight_3, np.array([-1,v1,v2]) )

        if gamma < 0:
            z =  1 - 1 / (1 + math.exp(gamma))
        else:
            z =  1 / (1 + math.exp(-gamma))

        predict = sgn(z)
        # print('\npredict: ',predict)
        # print('ans: ',y_train[i])

        if y_train[i] != predict:
            delta_3 = ( y_train[i] - z ) * z * ( 1 - z )
            delta_1 = y1 * (1 - y1) * ( np.dot(delta_3, weight_3) )
            delta_2 = y2 * (1 - y2) * ( np.dot(delta_3, weight_3) )

            weight_1 = weight_1 + learning_rate * np.dot(delta_1, x_train[i])
            weight_2 = weight_2 + learning_rate * np.dot(delta_2, x_train[i])
            weight_3 = weight_3 + learning_rate * np.dot(delta_3, np.array([-1,v1,v2]))

            train_error_rate = train_error_rate + 1


    training_accuracy = np.round( 1-(train_error_rate / iteration), 3 )
    print("======= Training =======")
    print('\nError: ', train_error_rate)
    print('Correct Rate: ', training_accuracy)
    print("========================")

    test_prediction = []
    for i in range(test.shape[0]):
        v1 = np.dot(weight_1, x_test[i])
        y1 = 1 / (1 + math.exp(-v1))
        v2 = np.dot(weight_2, x_test[i])
        y2 = 1 / (1 + math.exp(-v2))

        gamma = np.dot(weight_3, np.array([-1, v1, v2]))

        if gamma < 0:
            z = 1 - 1 / (1 + math.exp(gamma))
        else:
            z = 1 / (1 + math.exp(-gamma))

        predict = sgn(z)
        test_prediction.append(predict)
        # print('\npredict: ',predict)
        # print('ans: ',y_test[i])

        if y_test[i]!=predict:
            test_error_rate = test_error_rate + 1

    RMSE = rmse(np.array(test_prediction),y_test)
    print('RMSE: ',RMSE)

    test_accuracy = np.round( 1 - (test_error_rate / test.shape[0]), 3 )
    print("\n======= Testing =======")
    print('Error: ', test_error_rate)
    print('Correct Rate: ', test_accuracy)
    print("========================")


    return training_accuracy, test_accuracy, weight_1, weight_2, weight_3, RMSE

    # y1_point = []
    # y2_point = []
    # z_point = []
    # for i in range(train.shape[0]):
    #     v1 = np.dot(weight_1, x_train[i])
    #     y1 = 1 / (1 + math.exp(-v1))
    #     y1_point = np.append(y1_point, y1)
    #
    #     v2 = np.dot(weight_2, x_train[i])
    #     y2 = 1 / (1 + math.exp(-v2))
    #     y2_point = np.append(y2_point, y2)
    #
    #     gamma = np.dot(weight_3, np.array([-1, v1, v2]))
    #
    #     if gamma < 0:
    #         z = 1 - 1 / (1 + math.exp(gamma))
    #     else:
    #         z = 1 / (1 + math.exp(-gamma))
    #     z_point = np.append(z_point, z)

    # plotData_3D(train, y1_point, y2_point, z_point,weight_3)

# 畫圖
def plotData_3D(dataSet, y1_points, y2_points, z_points, weight):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # 畫training/testing data
    labels = dataSet[:, -1]
    idx_1 = np.where(dataSet[:, -1] == 1)
    p1 = ax.scatter3D(y1_points[idx_1], y2_points[idx_1], z_points[idx_1], marker='o', color='g', label=1, s=20)
    idx_2 = np.where(dataSet[:, -1] == 0)
    p2 = ax.scatter3D(y1_points[idx_2], y2_points[idx_2], z_points[idx_2], marker='x', color='r', label=0, s=20)

    X = np.arange(np.min(dataSet[:,1]), np.max(dataSet[:,1]), 0.1)
    Y = np.arange(np.min(dataSet[:,2]), np.max(dataSet[:,2]), 0.1)
    X, Y = np.meshgrid(X, Y)

    H = (np.min(z_points)+np.max(z_points))/2
    # Z = weight[1] * X + weight[2] * Y
    Z = ( (-1)*H - (weight[0] * X + weight[1] * Y) )/ weight[2]
    # z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.3)

    plt.show()

def spaceTransform(data,weight_1,weight_2):
    # y1_arr = [[] for _ in range(data.shape[0])]
    # y2_arr = [[] for _ in range(data.shape[0])]
    x_data = data[:, :data.shape[1] - 1]
    y1_arr = []
    y2_arr = []

    for i in range(data.shape[0]):
        v1 = np.dot(weight_1, x_data[i])
        y1 = 1 / (1 + np.exp(-v1))
        y1_arr.append(y1)

        v2 = np.dot(weight_2, x_data[i])
        y2 = 1 / (1 + np.exp(-v2))
        y2_arr.append(y2)

    return np.array(y1_arr), np.array(y2_arr)


# 畫圖
def plotData_2D(dataSet, weight_1, weight_2, weight_3, window, DataType, NoiseData):
    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(DataType + 'dataset')
    plt.xlabel('Y1')
    plt.ylabel('Y2')
    # plt.axis('equal')
    # ax.set_aspect('equal', adjustable='box')

    # 畫training/testing data
    labels = dataSet[:, -1]
    y1_point, y2_point = spaceTransform(dataSet, weight_1, weight_2)

    idx_1 = np.where(dataSet[:, -1] == 1)
    p1 = ax.scatter(y1_point[idx_1], y2_point[idx_1], marker='o', color='g', label=1, s=20)
    idx_2 = np.where(dataSet[:, -1] == 0)
    p2 = ax.scatter(y1_point[idx_2], y2_point[idx_2], marker='x', color='r', label=0, s=20)

    # print(len(NoiseData))
    # 畫雜點
    if len(NoiseData) > 0:
        p3 = ax.scatter(NoiseData[:, :, 1], NoiseData[:, :, 2], marker='^', color='b', label='Noise', s=20) # NoiseData is 3d-array


    # 畫分割線
    x = np.arange(np.min(dataSet[:,1]), np.max(dataSet[:,2]), 0.1)
    # # y1_line, y2_line = spaceTransform(dataSet, weight_1, weight_2)
    if weight_3[2]==0:
        y = 0
    else:
        y = (weight_3[0] - weight_3[1] * x) / weight_3[2]

    y1_line = []
    y2_line = []

    for i in range(x.shape[0]):
        v1 = np.dot(weight_1, (-1,x[i],y[i]))
        y1 = 1 / (1 + np.exp(-v1))
        y1_line.append(y1)
        v2 = np.dot(weight_2, (-1,x[i],y[i]))
        y2 = 1 / (1 + np.exp(-v2))
        y2_line.append(y2)

    ax.add_line(plt.Line2D(y1_line, y2_line))

    # 示意圖
    plt.legend(loc='upper right')
    # plt.show()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    if DataType == 'Training':
        canvas.get_tk_widget().place(x=450, y=80)

        # toolbar
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        toolbar.place(x=500, y=10)
    else:
        canvas.get_tk_widget().place(x=850, y=80)

        # toolbar
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        toolbar.place(x=900, y=10)



# if __name__ == '__main__':
    # data = readfile()
    # train, test = preprocess(data)
    # perceptron(train, test)
