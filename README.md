# Clickbait Highlighter

## Setup instructions

```linux
> git clone "https://github.com/asutosh29/clickbait-highligter-byop24"
> cd "clickbait-highligter-byop24"
```
## Make a new environment (Recommended)
1. virtualenv
```linux
> virtualenv env
> env\Scripts\activate
```
2. conda (recommended)
```linux
> conda create --name myenv python=3.12.8
> conda activate myenv
```
## Install the dependencies
```linux
> pip install requirements.txt
```
Also download the following folder from google drive link and put it in a folder named "ClickbaitModel" in the project directory "clickbait-highligter-byop24"
```linux
https://drive.google.com/drive/u/3/folders/1aWeD41L3elXf8jRPyRh3d1g7lRf8_Ggx
```
Folder structure should be like this: 
```
ClickbaitModel
- config.json
- pytorch_model.bin
- other files..
```

## Running the application

```python
> flask run --debug
```
This should start a server on local IP adress, navigate to the page url.

## Using the application
    1. Enter a text prompt (suggested around 10-30 words long)
    2. Click on [Get Score] button
    3. Wait for sometime while the UI updates.

