## §2 Backpropagation 推導與損失函數分析

### 2.1 符號定義與前向傳播 (Forward Pass)
設輸入矩陣為 $X \in \mathbb{R}^{m \times d}$（其中 $m$ 為批次大小/樣本數，$d$ 為特徵維度），令 $A^{(0)} = X$。
對於第 $l$ 層網路 ($l = 1, 2, \dots, L$)：
- 線性組合（未激活）：$Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}$
- 激活函數輸出：$A^{(l)} = \text{tanh}(Z^{(l)})$ （若為輸出層則依任務而定）
- 其中 $W^{(l)}$ 為權重矩陣，$b^{(l)}$ 為偏置向量。

### 2.2 輸出層誤差與梯度推導
定義損失函數對第 $L$ 層（輸出層）未激活值 $Z^{(L)}$ 的誤差訊號為 $\delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}}$。

1. **迴歸任務（MSE 損失）**：
   - 損失函數：$L_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$，其中輸出為線性輸出 $\hat{y} = Z^{(L)}$。
   - 誤差訊號：$\delta^{(L)} = \frac{\partial L_{\text{MSE}}}{\partial Z^{(L)}} = \frac{2}{m}(\hat{y} - y)$。

2. **二元分類任務（BCE 損失配合 Sigmoid 激活）**：
   - 損失函數：$L_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m [y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)]$，其中 $\hat{y} = \sigma(Z^{(L)}) = \frac{1}{1 + e^{-Z^{(L)}}}$。
   - 透過鏈鎖律（Chain Rule），其誤差訊號可簡化為：
     $$\delta^{(L)} = \frac{\partial L_{\text{BCE}}}{\partial Z^{(L)}} = \frac{1}{m}(\hat{y} - y)$$
     *(實作需特別注意對數的數值穩定性，避免 $\log(0)$ 發生)*。

對權重與偏置的偏導數為：
$$\frac{\partial L}{\partial W^{(l)}} = (A^{(l-1)})^T \delta^{(l)}$$
$$\frac{\partial L}{\partial b^{(l)}} = \sum_{i=1}^m \delta^{(l)}_i \quad \text{（對列加總）}$$

### 2.3 隱藏層誤差的反向傳遞（Chain Rule）
對於任意隱藏層 $l$，我們需要將輸出層的誤差透過鏈鎖律「再向前一層傳回」：
$$\delta^{(l)} = \frac{\partial L}{\partial Z^{(l)}} = \left( \frac{\partial L}{\partial Z^{(l+1)}} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \right) \odot \frac{\partial A^{(l)}}{\partial Z^{(l)}} = (\delta^{(l+1)} (W^{(l+1)})^T) \odot f'(Z^{(l)})$$
其中 $\odot$ 表示 Hadamard 乘積（元素相乘），$f'(Z^{(l)})$ 為隱藏層激活函數 $\text{tanh}$ 的導數（$1 - (A^{(l)})^2$）。
**原因說明**：三層網路之所以要將誤差再向前一層傳回，是因為網路具有複合函數結構。根據微積分的鏈鎖律，當前層的權重更新與誤差表現，直接受制於下一層的輸出誤差；唯有透過逐層遞迴相乘雅可比矩陣與權重轉置，才能正確將整體損失歸因（Credit Assignment）到較深或較淺的每一層參數上。

### 2.4 MSE 線性輸出與 BCE Sigmoid 輸出的比較
- **MSE 配合線性輸出**：假設輸出連續數值，其梯度與誤差呈線性正比（$\hat{y} - y$），適用於迴歸問題。當預測值離目標越遠，梯度越大；但在分類問題中使用時，搭配 sigmoid 會導致當預測極度錯誤時梯度趨近於零（梯度消失）。
- **BCE 配合 Sigmoid 輸出**：專為機率分類設計，透過對數項抵消了 sigmoid 導數在兩端飽和時的衰減，使得交叉熵損失在分類錯誤時能提供健康的梯度訊號，加速收斂。