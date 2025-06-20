# Clickbait Highlighter

## Setup instructions

```linux
> git clone "https://github.com/asutosh29/clickbait-highligter-byop24"
> cd "clickbait-highligter-byop24"
```
## Make a new environment (Recommended)
- conda (recommended) 
```linux
> conda env create -f environment.yml
> conda activate <env name in the .yml file>
```
## Install the dependencies
- install pip packages separately
```
> pip install -r reqPip.txt
```

Also download the following folder from google drive link and put it in a folder named "Clickbait1" in the project directory "clickbait-highligter-byop24"
```linux
https://drive.google.com/drive/u/3/folders/1aWeD41L3elXf8jRPyRh3d1g7lRf8_Ggx
```
Folder structure should be like this: 
```
clickbait-highligter-byop24
- Clickbait1
--- config.json
--- pytorch_model.bin
--- other files..
- models
- notebooks
...
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

