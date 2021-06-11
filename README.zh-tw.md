
<h1 align="center">口罩偵測與提醒系統</h1>

<div align="center">
    <strong>適合嵌入式系統的輕量化口罩正確配戴偵測</strong>
</div>

<div align="center">
    運用 Tensorflow/Keras 深度學習框架訓練模型、使用 Dash 前端框架、部署於 Google App Engine
</div>

<br/>

<div align="center">
    <!-- Python version -->
    <img src="https://img.shields.io/badge/python-v3.8-blue?style=flat-square"/>
    <!-- Last commit -->
    <img src="https://img.shields.io/github/last-commit/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Stars -->
    <img src="https://img.shields.io/github/stars/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Forks -->
    <img src="https://img.shields.io/github/forks/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Open issues -->
    <img src="https://img.shields.io/github/issues/achen353/Face-Mask-Detector?style=flat-square"/>
</div>

<br/>

<div align="center">
    <img src="./readme_assets/readme_cover.png"/>
</div>

<br/>

*閱讀 [英文](README.md) 版本 README.md。*

## 目錄
- [特色](#特色)
- [關於](#關於)
- [運用的框架與函式庫](#frameworkslibraries)
- [資料集](#資料集)
- [訓練結果](#訓練結果)
- [程式所需條件](#程式所需條件)
- [如何開始](#如何開始) 
- [如何使用](#如何使用)
- [Dash 程式展示](#Dash程式展示)
- [參考文獻與感謝](#參考文獻與感謝)
- [特許條款](#特許條款)

## 特色
- __輕量化模型：__  本專案所建立的 MFN 和 RMFD 模型分別只有 `2,422,339` 和 `2,422,210` 個參數。
- __偵測多個人臉：__ 能夠偵測一張圖或影格內多個人臉。
- __支援電腦攝影鏡頭：__ 能夠偵測照片及影像中的口罩配戴。
- __偵測非正確口罩：__ 能夠偵測非正確口罩配戴（例：下巴未包覆完全、鼻子露出、鼻子與嘴巴皆露出）。

## 關於
此程式能夠偵測圖片或影像中的人臉並分辨是否有正確配戴口罩。

在新冠肺炎疫情肆虐下，配戴口罩已被科學證實為有效阻斷病毒傳播的一種方式，也成為出入各大公共場所必須遵守的規定。在這樣的環境下，
此專案針對具有極高影像分類表現的輕量化 MobileNet V2 進行遷移學習以建立易部署且易結合嵌入式系統的開源口罩偵測模型提供公共運輸、娛樂場所、
醫院及其他營業場所參考使用。

另外由於大多數的開源口罩偵測模型僅能偵測是否有口罩配戴於臉上，此專案之模型能額外辨別是否有口鼻露出或使用其他物品遮住口鼻等非正確配戴狀況，
希望能夠近一步加強口罩配戴規定及實施。

## 運用的框架與函式庫
- __[OpenCV](https://opencv.org/) ：__ 用於處理影像及影格的電腦視覺函式庫
- __[OpenCV DNN Face Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py) ：__ 
  用於偵測人臉的 Caffe 深度學習 Single Shot-Multibox Detector（SSD）模型
- __[Tensorflow](https://www.tensorflow.org/) / [Keras](https://keras.io/) ：__ 用於建立此專案模型的主要深度學習框架
- __[MobileNet V2](https://arxiv.org/abs/1801.04381) ：__ 輕量化影像分類模型，
  在 ImageNet 影像分類上達到 71.3% Top-1 正確率及 90.1% Top-5 正確率
- __[Dash](https://plotly.com/dash/) ：__ 建立在 Plotly.js、React 和 Flask 之上的前端框架，此專案中用於程式展示

## 資料集
我們提供訓練於兩個不同資料集（RMFD 資料集和 MFN 資料集）的模型。

本專案的 RMFD 資料集取自 [口罩遮挡人脸数据集](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) 
並經過處理與採樣，而 MFN 資料集取自 [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) 和 
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset) 。

### RMFD 資料集
本資料集包括 __4,408__ 張圖片：
- `face_no_mask`： 2,204 張圖片
- `face_with_mask`： 2,204 張圖片

每張圖片皆為真實人臉，尺寸不一。我們將原始資料集內 90,568 張未戴口罩的圖片隨機採樣 2,204 張以建立 `face_no_mask`。 
`face_with_mask` 的數據則為原始資料集內所有配戴口罩的圖片。

### MFN 資料集
本資料集包括 __200,627__ 張圖片：
- `face_with_mask_correctly`： 67,193 張圖片
- `face_with_mask_incorrectly`： 66,899 張圖片
- `face_no_mask`： 66,535 張圖片

`face_with_mask_correctly` 及 `face_with_mask_incorrectly` 內圖片為 MaskedFace-Net 資料集內相對應類別內所有圖片縮放後之 128*128 影像。 
而 `face_no_mask` 則取自 Flickr-Faces-HQ Dataset (FFHQ)。 MaskedFace-Net 所有圖片內的口罩皆為電腦合成處理製成以產生足夠數據提供研究，
其中 `face_with_mask_incorrectly` 的 10% 為下巴露出之圖片、10% 為鼻子露出之圖片、80% 為口鼻皆露出之圖片.

### 下載資料集
資料集可以從這裡[下載](https://drive.google.com/file/d/1Y1Y67osv8UBKn_ANckCXPvY2aZqv1Cha/view?usp=sharing)。（2021/06/11）

## 訓練結果
兩個模型皆使用相對應資料集內 80% 的數據作為訓練集、20％ 作為驗證集兼測試集。 

MFN 模型                             |  RMFD 模型
:------------------------------------:|:--------------------------------------:
![](./figures/train_plot_MFN.jpg)   |  ![](./figures/train_plot_RMFD.jpg) 


我們發現 MFN 模型易將 `face_no_mask` 誤為 `face_with_mask_incorrectly`，即使這對提醒人配戴口罩影響不大（兩種結果皆會觸發提醒），
仍歡迎任何針對此模型表現的改進建議。

## 程式所需條件
此專案使用 Python 3.8 運行於 MacOS Big Sur 11.1. 模型的訓練使用雲端的 GCP 
Compute Engine（8 vCPUs, 13.75 GB memory）配備 `tensorflow==2.4.0`。所有依賴與所需函式庫皆列於`requirements.txt`。

備註：由於模型建立時 `opencv-python==4.4.0` 的 `cv2.imshow` 無法在 MacOS Big Sur 上使用 [無法在 MacOS Big Sur 上使用](https://github.com/skvark/opencv-python/issues/423) 
，我們改用 `opencv-python-headless==4.5.1`。然而最新版本 `opencv-python==4.5.1.48` 似乎已將問題修復，若你已安裝 `opencv-python`，
則只需升級該函式庫即可。

## 如何開始
1. 開啟終端，用 `cd` 進到你想要存放此專案的資料夾並複製此儲存庫：
```
$ git clone https://github.com/achen353/Face-Mask-Detector.git
```
2. [下載並安裝](https://docs.conda.io/en/latest/miniconda.html) Miniconda。
3. 建立具有 `requirements.txt` 所有函式庫之環境：
```
$ conda create --name env_name --file requirements.txt
```
4. 現在你可以用 `cd` 進到複製的專案資料夾裡執行程式或檢視檔案.

## 如何使用

### 偵測照片裡的人臉及口罩
使用 `cd` 進入 `/src/` 資料夾並執行：
```
$ python detect_mask_images.py -i <image-path> [-m <model>] [-c <confidence>]
```

### 偵測電腦鏡頭影像裡的人臉及口罩
使用 `cd` 進入 `/src/` 資料夾並執行：
```
$ python detect_mask_video.py [-m <model>] [-c <confidence>]
```

### 用提供的資料集自己訓練模型
使用 `cd` 進入 `/src/` 資料夾並執行：
```
$ python train.py [-d <dataset>]
```
記得修改 `train.py` 裡的路徑設定以避免覆寫提供之已訓練模型。

Note: 
- `<image-path>` 應為對於專案根目錄（而非 `/src/`）之相對路徑。
- `<model>` 應為 `str` 字串，可接受字串有 `MFN` 及 `RMFD`。預設為 `MFN`。
- `<confidence>` 應為 `float` 浮點數，可接受數值介於 `0` and `1` 之間（包含），預設為 `0.5`。
- `<dataset>` 應為 `str` 字串，可接受字串有 `MFN` 及 `RMFD`，預設為 `MFN`。

## Dash 程式展示
本專案之程式展示 [在這裡](https://face-mask-detection-300106.wl.r.appspot.com)，目前仍在測試階段。

### 在你的電腦上執行 Dash 程式展示
1. 將 `app.run_server(host='0.0.0.0', port=8080, debug=True)` 修改為 `app.run_server(debug=True)`：
2. 執行下列指令：
```
$ python main.py
```
3. 在你的瀏覽器網址列上輸入 `http://127.0.0.1:8050/` 以透過 Dash app 的預設端口使用程式。若是端口正在被使用中，你可以更改端口。

## 參考文獻與感謝
- 口罩遮挡人脸数据集（[Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) ，RMFD）
- Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of 
  correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, 
  Elsevier, 2020, [DOI:10.1016/j.smhl.2020.100144](https://doi.org/10.1016/j.smhl.2020.100144)
- Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection 
  mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", 
  CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 
  2020, [DOI:10.32604/cmes.2020.011663](DOI:10.32604/cmes.2020.011663)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
- [Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Object Detection](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection)

## 特許條款
[MIT © Andrew Chen](https://github.com/achen353/Face-Mask-Detector/blob/master/LICENSE)
