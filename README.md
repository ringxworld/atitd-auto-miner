# ATITD Auto Miner

Fully automatic but very brittle. Enjoy!
 
### Installation:

Miniconda/Anaconda setup

Installing on windows:
https://docs.conda.io/en/latest/miniconda.html
```commandline
conda init
conda create env --name ATITDScripts
conda activate ATITDScripts
conda install pip
cd /path/to/repo/root/
pip install -r requirements.txt
python main.py --help
```

### Usage
Start with F8F8 Camera zoomed in about halfway like this

![Example Picture](https://i.gyazo.com/909935b73b056ee77cd6ab0a49b32753.jpg)

Press ALT + L ingame to lock camera to this position

with a command line terminal open run the script
```commandline
cd /path/to/repo/root
python main.py --clusters 7 --downsample 3 --eps 1 --min_samples 40
```

### Troubleshooting

include --debug flag for FPS and seeing what the script views as foreground pixels

```commandline
python main.py --clusters 7 --downsample 3 --eps 1 --min_samples 40 --debug
```

If the default bounds being shown with --debug aren't fitting all the nodes. 

Zoom out or change the bounds its searching for foreground pixels in:

```commandline
Change bounds to the length of your window that you wish to screen cap

python main.py --clusters 7 --downsample 3 --eps 1 --min_samples 40 --bounds 200 500 950 740

```

#### Additional Troubleshooting

For full list of supported arguments type:

```commandline
python main.py --help
```
