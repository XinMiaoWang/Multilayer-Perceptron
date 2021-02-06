# Multilayer-Perceptron
### 實作多層感知機  
* 程式執行說明：  
1、點擊選擇檔案，可選擇dataset。  
2、學習率設定可輸入大於0，小於等於1的小數。  
3、收斂條件設定，即輸入感知機疊代次數。  
4、完成1~3設定後，點擊「執行」即可顯示訓練準/測試準確率、RMSE、鍵結值和訓練/測試資料分布圖。  
5、訓練進度條可顯示目前訓練進度。  
6、Toolbar可調整分布圖大小等等。  

![1](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/1.PNG)  

* 程式碼簡介：  
  - GUI設計  
    createWidgets()函式會建立出使用者介面，包含文字、輸入框、按鈕等等。  
    進度條設計，會隨訓練次數的增加更新訓練進度。  
      
    ![2](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/2.png)  

  - 資料前處理  
    因為此感知機設計只能進行二分類，再加上每個dataset要分類的label都不相同，因此在資料前處理的時候統一了各個dataset的label，label數字較大的標為1代表正樣本，較小的標為0代表負樣本。  
      
    ![3](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/3.png)  
      
  - 多層感知機設計  
    步驟一：決定架構  
    預設為二層，隱藏層神經元數目為2。  
      
    ![4](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/4.png)  

    步驟二：網路初始化  
    Weight的初始值為隨機產生，範圍在[0,1)之間。  
    ![5](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/5.png)  

    步驟三：前饋階段  
    - 計算公式：  
     ![6](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/6.PNG)  
     ![7](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/7.png)  
    - 分類條件：  
      輸出結果大於等於0.5即分類為正樣本，反之則分類為負樣本。  
     ![8](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/8.png)  
     ![9](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/9.png)  
       
    步驟四：倒傳遞階段  
    - 如果第 j 個類神經元是輸出層的類神經元：  
     ![10](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/10.PNG)  
    - 如果第 j 個類神經元是隱藏層的類神經元：  
     ![11](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/11.PNG)  

    步驟五：調整鍵結值向量  
    判斷資料是否有正確分類，若誤判就對Weight進行更新，根據以下原則來計算新的Weight：  
     ![12](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/12.PNG)  

    步驟六：收斂條件測試  
      
    步驟二到步驟五反覆執行，直到達到我們所設的收斂條件(疊代次數)才停止(每一次疊代訓練一筆資料)。  
      
    評估指標 – 均方根誤差(RMSE)  
    - 計算公式：  
     ![13](https://github.com/XinMiaoWang/Multilayer-Perceptron/blob/main/demo/13.PNG)  

* 繪圖  
  使用matplotlib套件。圈圈為正樣本，叉叉為負樣本，藉由訓練完成的感知機可以得到最終的Weight_1、Weight_2和Weight_3，根據Weight_1和Weight_2我們可以對原始座標(X,Y)進行空間轉換到(Y1，Y2)，再由Weight_3畫出分割正負樣本的圖形。






