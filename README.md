<h2>ImageMask-Dataset-BCData  (2024/07/03)</h2>

This is ImageMask Dataset forImageMask Dataset for BCData: A Large-Scale Dataset and Benchmark for Cell Detection and Counting
(MICCAI 2020).
<br>
The original dataset used here has been taken from the following web-site<br>
<b>BCData: A Large-Scale Dataset and Benchmark for Cell Detection and Counting </b><br>
<pre>
https://sites.google.com/view/bcdataset
</pre>
The BCData data can be downloaded from the google drive 
<a href="https://drive.google.com/file/d/16W04QOR1E-G3ifc4061Be4eGpjRYDlkA/view?usp=sharing">BCData.zip</a>.


<b>Download ImageMask-Dataset</b><br>
You can download the BCData ImageMask-Dataset, derived by us from the original BCData, from the google drive 
<a href="https://drive.google.com/file/d/1J62EHi0JLDfL-lVPkN_W7KXaeOEVOrWC/view?usp=sharing">
BCData-ImageMask-Dataset-V1.zip</a>
<br>


<h3>1. Dataset Citation</h3>

<a href="https://link.springer.com/chapter/10.1007/978-3-030-59722-1_28">
BCData: A Large-Scale Dataset and Benchmark for Cell Detection and Counting</a><br>
Medical Image Computing and Computer Assisted Intervention – MICCAI 2020, 2020, Volume 12265<br>
ISBN : 978-3-030-59721-4<br>

Zhongyi Huang, Yao Ding, Guoli Song, Lin Wang, Ruizhe Geng, Hongliang He, Shan Du, Xia Liu, <br>
Yonghong Tian, Yongsheng Liang, S. Kevin Zhou & Jie Chen<br>


<h3>2. ImageMaskDataset Generation</h3>

If you would like to create BCData ImageMask Dataset by yourself,
please download the set from the google drive 
<a href="https://drive.google.com/file/d/16W04QOR1E-G3ifc4061Be4eGpjRYDlkA/view?usp=sharing">BCData.zip</a>.

<br>

Please run the following command for Python script <a href="./ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>.
<br>
<pre>
> python ImageMaskDatasetGenerator.py
</pre>
This command will generate 640x640 JPG image mask dataset.<br>
<pre>
./BCData-ImageMask-Dataset-V1
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─validation
    ├─images
    └─masks
</pre>

<hr>
Train images sample<br>
<img src="./asset/train_images_sample.png" width=1024 heigh="auto"><br><br>
Train mask sample<br>
<img src="./asset/train_masks_sample.png" width=1024 heigh="auto"><br><br>


Dataset Statistics <br>
<img src="././BCData-ImageMask-Dataset-V1_Statistics.png" width="512" height="auto"><br>
