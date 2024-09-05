# BoutRightYOLO

# Installation
Download anaconda
https://www.anaconda.com/download


# Getting Started
Follow the following steps to get started with boutRight:

1. **Clone the repository**

Clone the repository using Git, if you have a git installed in your computer. To download git go to [Downloads](https://git-scm.com/downloads) and choose a git suitable for your operating system. You can learn more about git [here](https://www.freecodecamp.org/news/git-and-github-for-beginners/).
Optionally, you can install git in anaconda prompt by typing the following command:

```
conda install git
```

Once the git download is done, open a terminal or command prompt and run the following command:

```
git clone https://github.com/RuthKassahun/BoutRightYOLO.git
```

If you don't have a git installed, alternatively, you can download the zip file of this repository if you haven't installed git on your local machine.

2. ***Navigate to the Project Folder***
   
Open Anaconda Prompt and navigate to the "boutRight" folder by using the cd command. Replace ***/path/to/boughtRight_yolo*** with the actual path to the "boutRight" folder on your local machine.
```
cd /path/to/boutRight_yolo
```
3. ***Create a Virtual Environment***

Create a virtual environment using the provided environment.yml file. This YAML file specifies the dependencies and environment settings required for the project.

```
conda env create -f boutright_environment.yml
```
4. ***Activate the Environment***

Activate the newly created conda environment 
```
conda activate boutRight
```
Incase you want to do some sanity check if pytorch works with the configured cuda version 
type `python` once you can see these `>>>` type

```import pytorch```

```torch.cuda.get_device_name(0)``` -- This will display the GPU device your cuda is currently accessing

## Pre-Processing Dataset (Generate Spectrogram from song file)
place all your song files inside `songs_for_training` folder in the main `boutRight_yolo` folder

1. Prepare a virtual envrinoment for labelImg
   
   1.1 Create a new conda envirnoment for annotating spectograms using `labelImg`. As LabelImg runs in python 3.8
    ```
    conda create -n imageannotation python==3.8
    ```
   1.2 Activate the envirnoment
    ```
    conda activate imageannotation
    ```
    1.3 Install LabelImg
    ```
    pip install labelImg
    ```

3. Update your `data.yaml` configuration file

   Open data.yaml file and change the directory path for both train and val.
   ```
   train : {\path\to\boutRight_yolo}\dataset\images\train
   val :   {\path\to\boutRight_yolo}\dataset\val
   ```
   for example : if you saved your boutRight_yolo folder in a the following path `D:\Code\boutRight_yolo`, you can change it as
   ```
   train :   D:\Code\boutRight_yolo\dataset\images\train
   val :     D:\Code\boutRight_yolo\dataset\images\val
   ```

4. Open boutRight_yolo_train.ipynb jupyter notebook in visual studio code or in localhost
   This will go through all wav files in the given `songs_for_training` folder and create a new folder named `images_for_training` inside your main directory and add all the generated spectrograms.
   Run the first two cells

6. Open LabelImg
   Inside labelImg conda envirnoment type
   ```
   labelImg
   ```
   This will open a small window to annotate the spectrograms

7. Place all your annotation files inside `annotations` folder in the main directory, then run the third cell (Split Dataset section) from the notebook. 

   
   

   
   



