# HW01 程式碼執行說明

# HW01: MLP from Scratch (NumPy)

本專案使用純 NumPy 從零實作多層感知機 (MLP)，支援任意隱藏層結構、前向傳播、手動反向傳播 (Backward Pass)、數值梯度檢查、迴歸任務與二元分類任務。

---

## 1. 環境需求與版本
- **Python 版本**: Python 3.10.x（實際開發與測試環境：Python 3.10）
- **核心依賴套件（精確版本）**:
  - `numpy==2.2.6`
  - `matplotlib==3.10.9`

安裝指令：
```bash
pip install -r ../requirements.txt


## 2. 隨機種子與參數設定 (Reproducibility)
- **資料集隨機種子**: 訓練集 = 42, 驗證集 = 100, 測試集 = 200
- **模型權重初始化**: Xavier 初始化 ($\text{fan\_in}$)
- **雙螺旋資料雜訊 (Noise)**: $0.035$

## 3. 執行順序
請依序執行以下指令以產生所有模型與圖表：
1. `py generate_data.py` (資料生成模組)
2. `py mlp.py` (MLP 核心架構與 `predict()` 方法)
3. `py gradient_check.py` (梯度檢查與相對誤差驗證，輸出 `gradient_check.png`)
4. `py train_regression.py` (非線性函數逼近與迴歸評估，輸出 `loss_curves.png`, `regression_comparison.png`)
5. `py train_spiral.py` (雙螺旋二元分類訓練與評估，輸出 `spiral_boundaries.png`)
6. `py compare_hyperparam.py` (學習率超參數對比，輸出 `hyperparam_comparison.png`)