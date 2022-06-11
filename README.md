# ATITD Auto Miner

Fully automatic but very brittle. Enjoy!
 
### Installation:

Miniconda/Anaconda setup

Installing on windows:
https://docs.conda.io/en/latest/miniconda.html
```commandline
conda create -n ATITDScripts
conda activate ATITDScripts
conda install pip
cd /path/to/repo/root/
pip install -r requirements.txt
python setup.py develop
run_miner --help
```

### Usage
Start with F8F8 Camera zoomed in about halfway like this

![Example Picture](https://i.gyazo.com/909935b73b056ee77cd6ab0a49b32753.jpg)

Press ALT + L ingame to lock camera to this position

with a command line terminal open run the script
```commandline
cd /path/to/repo/root
run_miner --clusters 7 --downsample 3 --eps 1 --min_samples 40
```

### Troubleshooting

include --debug flag for FPS and seeing what the script views as foreground pixels

```commandline
run_mine --debug
```

If the default bounds being shown with --debug aren't fitting all the nodes. 

Zoom out or change the bounds its searching for foreground pixels in:

```commandline
Change bounds to the length of your window that you wish to screen cap

Working runtimeparams, change cluster count depending on clusters per rock type
run_miner --clusters 7 --wait_frames 100 --downsample 4 --eps 1 --min_samples 750 --run_ocr --four_combinations --debug

```

#### Additional Troubleshooting

For full list of supported arguments type:

```commandline
run_miner --help
```
