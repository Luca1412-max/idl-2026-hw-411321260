# HW01 深度學習作業實驗報告

## §1 摘要 (Abstract)
本作業旨在透過純 NumPy 自行實作多層感知機 (MLP)，從零建構前向傳播 (Forward Pass) 與反向傳播 (Backward Pass) 機制。實驗涵蓋兩大核心任務：一是「雙變數非線性函數逼近 (Regression)」，二是「雙螺旋二元分類 (Classification，noise=0.035)」。透過嚴格固定隨機種子與超參數，並使用 Xavier 初始化與數值梯度檢查（$h = 1e-5$）驗證梯度正確性，本報告系統性評估了 2 層 (`[32]`) 與 3 層 (`[32, 32]`) MLP 在不同模型容量下的擬合表現、損失收斂軌跡與決策邊界特徵。

---

## §2 Backpropagation 推導與損失函數分析

### 2.1 符號定義與前向傳播 (Forward Pass)
設輸入矩陣為 $X \in \mathbb{R}^{m \times d}$（其中 $m$ 為批次大小/樣本數，$d$ 為特徵維度），令 $A^{(0)} = X$。
對於第 $l$ 層網路 ($l = 1, 2, \dots, L$)：
- **矩陣與變數形狀定義**：
  - 輸入/啟動值：$A^{(l-1)} \in \mathbb{R}^{m \times d_{l-1}}$（第 $0$ 層即為輸入 $X \in \mathbb{R}^{m \times d}$）
  - 權重矩陣：$W^{(l)} \in \mathbb{R}^{d_{l-1} \times d_l}$
  - 偏置向量：$b^{(l)} \in \mathbb{R}^{1 \times d_l}$（廣播至 $m$ 筆樣本）
  - 線性組合（未激活）：$Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)} \in \mathbb{R}^{m \times d_l}$
  - 激活函數輸出：$A^{(l)} = \text{tanh}(Z^{(l)}) \in \mathbb{R}^{m \times d_l}$（隱藏層）或 $\hat{y} \in \mathbb{R}^{m \times d_{out}}$（輸出層）
  - 誤差訊號：$\delta^{(l)} = \frac{\partial L}{\partial Z^{(l)}} \in \mathbb{R}^{m \times d_l}$

### 2.2 輸出層誤差與梯度推導
定義損失函數對第 $L$ 層（輸出層）未激活值 $Z^{(L)}$ 的誤差訊號為 $\delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}}$。

1. **迴歸任務（$\frac{1}{2}\text{MSE}$ 損失，本實作採用 $\frac{1}{2}$ 係數以簡化梯度計算）**：
   - 損失函數：$L_{\text{MSE}} = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$，其中輸出為線性輸出 $\hat{y} = Z^{(L)}$。
   - 誤差訊號：$\delta^{(L)} = \frac{\partial L_{\text{MSE}}}{\partial Z^{(L)}} = \frac{1}{m}(\hat{y} - y)$。

2. **二元分類任務（BCE 損失配合 Sigmoid 激活的完整鏈鎖律推導）**：
   - 損失函數：$L_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$，其中預測輸出為 $\hat{y} = \sigma(Z^{(L)}) = \frac{1}{1 + e^{-Z^{(L)}}}$。
   - 根據微積分鏈鎖律（Chain Rule），輸出層未激活值 $Z^{(L)}$ 的誤差訊號 $\delta^{(L)}$ 可拆解為損失對預測值的偏導數與 Sigmoid 導數的相乘：
     $$\delta^{(L)} = \frac{\partial L_{\text{BCE}}}{\partial Z^{(L)}} = \frac{\partial L_{\text{BCE}}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z^{(L)}}$$
   - 首先計算損失函數 $L_{\text{BCE}}$ 對預測輸出 $\hat{y}$ 的偏導數：
     $$\frac{\partial L_{\text{BCE}}}{\partial \hat{y}} = -\frac{1}{m} \left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right) = -\frac{1}{m} \left( \frac{y(1-\hat{y}) - \hat{y}(1-y)}{\hat{y}(1-\hat{y})} \right) = -\frac{1}{m} \left( \frac{y - \hat{y}}{\hat{y}(1-\hat{y})} \right) = \frac{\hat{y} - y}{m\hat{y}(1-\hat{y})}$$
   - 接著計算 Sigmoid 激活函數對未激活值 $Z^{(L)}$ 的導數 $\sigma'(Z^{(L)})$：
     $$\sigma'(Z^{(L)}) = \sigma(Z^{(L)})(1 - \sigma(Z^{(L)})) = \hat{y}(1-\hat{y})$$
   - 將兩者相乘，使分母的 $\hat{y}(1-\hat{y})$ 與導數完美相消：
     $$\delta^{(L)} = \frac{\hat{y} - y}{m\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \frac{1}{m}(\hat{y} - y)$$
     *(實作需特別注意對數的數值穩定性，加上微小量 $\epsilon = 1e-15$ 避免 $\log(0)$ 發生)*。

對權重與偏置的偏導數為：
$$\frac{\partial L}{\partial W^{(l)}} = (A^{(l-1)})^T \delta^{(l)}$$
$$\frac{\partial L}{\partial b^{(l)}} = \sum_{i=1}^m \delta^{(l)}_i \quad \text{（對行加總）}$$

### 2.3 隱藏層誤差的反向傳遞（Chain Rule）與梯度回傳原因
對於任意隱藏層 $l$，我們需要將輸出層的誤差透過鏈鎖律「再向前一層傳回」：
$$\delta^{(l)} = \frac{\partial L}{\partial Z^{(l)}} = \left( \frac{\partial L}{\partial Z^{(l+1)}} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \right) \odot \frac{\partial A^{(l)}}{\partial Z^{(l)}} = (\delta^{(l+1)} (W^{(l+1)})^T) \odot f'(Z^{(l)}$$
- 其中 $\odot$ 表示 Hadamard 乘積（元素相乘），$f'(Z^{(l)})$ 為隱藏層激活函數 $\text{tanh}$ 的導數（$1 - (A^{(l)})^2$）。
- **原因說明**：三層或多層網路之所以要將誤差逐層向前傳回，是由於神經網路具備複合函數結構。根據微積分的鏈鎖律，當前層的權重更新與誤差貢獻，直接受制於下一層的輸出誤差；唯有透過逐層遞迴相乘權重轉置與激活函數導數，才能正確將整體損失歸因（Credit Assignment）到較深或較淺的每一層參數上。

### 2.4 MSE 線性輸出與 BCE Sigmoid 輸出的比較
- **MSE 配合線性輸出**：假設輸出連續數值，其梯度與誤差呈線性正比（$\hat{y} - y$），適用於迴歸問題。當預測值離目標越遠，梯度越大；但在分類問題中使用時，搭配 sigmoid 會導致當預測極度錯誤時其導數趨近於零（發生梯度飽和與消失）。
- **BCE 配合 Sigmoid 輸出**：專為機率分類設計，透過對數項完美抵消了 sigmoid 導數在兩端飽和時的衰減，使得交叉熵損失在分類錯誤時能提供健康的梯度訊號，加速收斂。

---

## 三、 梯度檢查與參數總數分析

### 3.1 參數總數計算與形狀表
各架構的參數個數計算方式如下（假設輸入維度 $d_{in}=2$, 輸出維度 $d_{out}=1$）：
1. **兩層 MLP (`2-32-1`) 架構**：
   - 第一層權重 $W^{(1)} \in \mathbb{R}^{2 \times 32}$ 與偏置 $b^{(1)} \in \mathbb{R}^{1 \times 32}$：$(2 \times 32) + 32 = 96$
   - 輸出層權重 $W^{(2)} \in \mathbb{R}^{32 \times 1}$ 與偏置 $b^{(2)} \in \mathbb{R}^{1 \times 1}$：$(32 \times 1) + 1 = 33$
   - **參數總數**：$96 + 33 = 129$ 個參數。

2. **三層 MLP (`2-32-32-1`) 架構**：
   - 第一隱藏層 $W^{(1)} \in \mathbb{R}^{2 \times 32}, b^{(1)} \in \mathbb{R}^{1 \times 32}$：$(2 \times 32) + 32 = 96$
   - 第二隱藏層 $W^{(2)} \in \mathbb{R}^{32 \times 32}, b^{(2)} \in \mathbb{R}^{1 \times 32}$：$(32 \times 32) + 32 = 1056$
   - 輸出層 $W^{(3)} \in \mathbb{R}^{32 \times 1}, b^{(3)} \in \mathbb{R}^{1 \times 1}$：$(32 \times 1) + 1 = 33$
   - **參數總數**：$96 + 1056 + 33 = 1185$ 個參數。

### 3.2 梯度檢查 (`gradient_check.py`) 結果與逐層驗證表 (`np.random.seed(0)`)
使用微小量 $h = 1e-5$ 進行數值梯度與解析梯度的相對誤差比對。各模型架構的逐層參數數量與相對誤差整理如下表：

| 任務類型 | 模型架構 (隱藏層) | 網路層別 | 參數矩陣形狀 | 該層參數數量 | 數值相對誤差 (Rel Error, $h=1e-5$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **分類** | 2層 MLP (`[32]`) | Layer 1 (Hidden) | $(2 \times 32) + 32$ | 96 | $\approx 3.66 \times 10^{-10}$ |
| **分類** | 2層 MLP (`[32]`) | Layer 2 (Output) | $(32 \times 1) + 1$ | 33 | $\approx 9.41 \times 10^{-11}$ |
| **分類** | 3層 MLP (`[32, 32]`) | Layer 1 (Hidden 1) | $(2 \times 32) + 32$ | 96 | $\approx 3.58 \times 10^{-10}$ |
| **分類** | 3層 MLP (`[32, 32]`) | Layer 2 (Hidden 2) | $(32 \times 32) + 32$ | 1024 | $\approx 4.98 \times 10^{-10}$ |
| **分類** | 3層 MLP (`[32, 32]`) | Layer 3 (Output) | $(32 \times 1) + 1$ | 33 | $\approx 8.98 \times 10^{-11}$ |
| **迴歸** | 2層 MLP (`[32]`) | Layer 1 (Hidden) | $(2 \times 32) + 32$ | 96 | $\approx 7.47 \times 10^{-11}$ |
| **迴歸** | 2層 MLP (`[32]`) | Layer 2 (Output) | $(32 \times 1) + 1$ | 33 | $\approx 5.68 \times 10^{-12}$ |
| **迴歸** | 3層 MLP (`[32, 32]`) | Layer 1 (Hidden 1) | $(2 \times 32) + 32$ | 96 | $\approx 1.15 \times 10^{-10}$ |
| **迴歸** | 3層 MLP (`[32, 32]`) | Layer 2 (Hidden 2) | $(32 \times 32) + 32$ | 1024 | $\approx 6.76 \times 10^{-11}$ |
| **迴歸** | 3層 MLP (`[32, 32]`) | Layer 3 (Output) | $(32 \times 1) + 1$ | 33 | $\approx 1.23 \times 10^{-11}$ |

- **梯度異常排查方法**：若檢查時發現相對誤差大於 $1 \times 10^{-5}$，通常排查步驟為檢查前向傳播偏置、矩陣轉置方向及數值擾動重置。

---

## 四、 實驗總覽與四模型彙整表
本作業透過純 NumPy 自行實作多層感知機 (MLP)，分別在「雙變數非線性函數逼近 (Regression)」與「雙螺旋二元分類 (Classification，noise=0.035)」兩大任務上進行 2 層 (`[32]`) 與 3 層 (`[32, 32]`) 的效能評估。（註：迴歸任務之 Loss 採用 $\frac{1}{2}\text{MSE}$ 定義）。

| 任務類型 | 模型架構 (隱藏層) | 訓練超參數 | 訓練集表現 | 驗證集表現 | 獨立測試集表現 (MSE / Accuracy) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **迴歸 (Regression)** | 2層 MLP (`[32]`) | LR=0.2, 1000 ep | $\frac{1}{2}\text{MSE}: 0.0315$ | $\frac{1}{2}\text{MSE}: 0.0312$ | **$\frac{1}{2}\text{MSE}: 0.0310$** |
| **迴歸 (Regression)** | 3層 MLP (`[32, 32]`) | LR=0.2, 1000 ep | $\frac{1}{2}\text{MSE}: 0.0028$ | $\frac{1}{2}\text{MSE}: 0.0029$ | **$\frac{1}{2}\text{MSE}: 0.0030$** |
| **分類 (Classification)**| 2層 MLP (`[32]`) | LR=0.5, 1500 ep | BCE: 0.012 | BCE: 0.015 | **Acc: 99.80%** |
| **分類 (Classification)**| 3層 MLP (`[32, 32]`) | LR=0.5, 1500 ep | BCE: 0.003 | BCE: 0.004 | **Acc: 100.00%** |

---

## 五、 任務一：雙變數非線性函數逼近實驗 (Regression)
- **實驗數據與圖表**：對應圖表為 **`hw01/figures/loss_curves.png`** 與 **`hw01/figures/regression_comparison.png`**（包含 5 個子圖：真實曲面、兩層預測曲面、三層預測曲面、兩層絕對誤差曲面、三層絕對誤差曲面）。
- **分析觀察**：
  1. 在透過驗證集調參（LR=0.2, 1000 epochs）後，三層 MLP 測試 $\frac{1}{2}\text{MSE}$ 達 $0.0030$，顯著優於兩層 MLP 的 $0.0310$（提升約 10 倍）。
  2. 從 `regression_comparison.png` 的 3D 網格曲面與兩張獨立誤差曲面可知，三層模型的預測曲面與真實曲面高度吻合，絕對誤差大幅降低。
  3. **參數規模與容量辯證**：三層模型（1185 參數）相較兩層模型（129 參數）多出近 9 倍參數，在複雜非線性函數逼近上展現出顯著的階層化特徵萃取優勢。

---

## 六、 任務二：雙螺旋二元分類實驗 (Classification, noise=0.035)
- **實驗數據與圖表**：對應圖表為 `hw01/figures/spiral_boundaries.png` 及 `loss_curves.png` 右圖。混淆矩陣實測數據如下：
  - 2層 MLP (`[32]`)：$\text{Accuracy} = 99.80\%$，混淆矩陣 $\text{TP}=249, \text{TN}=250, \text{FP}=0, \text{FN}=1$。
  - 3層 MLP (`[32, 32]`)：$\text{Accuracy} = 100.00\%$，混淆矩陣 $\text{TP}=250, \text{TN}=250, \text{FP}=0, \text{FN}=0$。
- **分析觀察**：
  1. 雙螺旋分類在 LR=0.5、1500 epochs 設定下，兩層與三層 MLP 分別達到 $99.80\%$ 與 $100.00\%$ 的準確率。
  2. 超參數對比實驗（`hyperparam_comparison.png`）顯示，LR=0.5 在約 250 輪時驗證 Accuracy 即接近 1.0，而 LR=0.1 與 1.0 需要 600-800 輪以上，證實學習率選擇對優化動態具決定性影響。

---

## 七、 模型比較與反思總結 
1. **參數規模與效能之權衡**：實驗證實三層 MLP（1185 參數）在迴歸與分類任務中皆展現出極高的擬合上限，但其參數量為兩層 MLP（129 參數）的近 9 倍。
2. **優化難度與網路深度的辯證**：深層網路的損失地景更具非凸性。若缺乏適當的超參數（如 1500 epochs 與 lr=0.5），深層模型極易陷入區域極小值；反之，若超參數契合，則能展現出壓倒性的特徵階層化萃取能力（分類達 100% 準確率，超參數實驗中 LR=0.5 在 250 輪內驗證準確率即飆升至 0.99 以上）。
3. 實驗嚴格遵守驗證集調參規範，確保評估指標客觀性。

---

## 八、 AI 使用宣告與聲明
- **使用工具**：人工智慧輔助工具 (Gemini)。
- **使用目的**：協助撰寫程式架構、除錯與數學推導梳理。
- **驗證方式**：所有程式碼與數據皆經人工逐一核對與驗證 (Human-Judgment)。