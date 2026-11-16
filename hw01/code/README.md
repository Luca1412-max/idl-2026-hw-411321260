# HW01 程式碼執行說明

# HW01: MLP from Scratch (NumPy)

本專案使用純 NumPy 從零實作多層感知機 (MLP)，支援任意隱藏層結構、前向傳播、手動反向傳播 (Backward Pass)、數值梯度檢查、迴歸任務與二元分類任務。

---

## 1. 環境需求與版本
- **Python 版本**: Python 3.10.x（實際開發與測試環境：Python 3.10）
- **核心依賴套件（精確版本）**:
  - `numpy==2.2.6`
  - `matplotlib==3.10.9`
  - `scipy==1.15.2`

安裝指令：
```bash
# 透過 requirements.txt 批次安裝
pip install -r ../requirements.txt

# 或直接安裝各套件精確版本
pip install numpy==2.2.6 matplotlib==3.10.9 scipy==1.15.2


## 2. 隨機種子與參數設定 (Reproducibility)
- **模型權重初始化**: Xavier 初始化 ($\text{fan\_in}$)
- **雙螺旋分類任務資料分割與種子**:
  - 訓練集種子：`42` (1000 筆/類別)
  - 驗證集種子：`100` (250 筆/類別)
  - 測試集種子：`200` (250 筆/類別)
  - 雙螺旋資料雜訊 (Noise): $0.035$[cite: 22]
- **迴歸任務資料分割與種子**:
  - 資料生成種子：`42` (共 3000 筆)
  - 切分方式：前 2000 筆為訓練集，2000-2500 筆為驗證集，2500-3000 筆為測試集。

## 3. 執行順序與輸出路徑
請在 `hw01/code/` 目錄下依序執行以下指令以產生所有模型與圖表：
1. `py generate_data.py` (資料生成核心模組，供訓練腳本匯入)
2. `py mlp.py` (MLP 核心架構、前向/反向傳播與 `predict()` 方法)
3. `py gradient_check.py` (梯度檢查與相對誤差驗證)
   - **輸出路徑**：`../figures/gradient_check.png`
4. `py train_regression.py` (非線性函數逼近與迴歸評估)
   - **輸出路徑**：`../figures/loss_curves.png`, `../figures/regression_comparison.png`
5. `py train_spiral.py` (雙螺旋二元分類訓練與評估)
   - **輸出路徑**：`../figures/spiral_boundaries.png`
6. `py compare_hyperparam.py` (學習率超參數對比實驗)
   - **輸出路徑**：`../figures/hyperparam_comparison.png`