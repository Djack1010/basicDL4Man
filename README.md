## Setup environment
Install `virtualenv`
```
sudo apt install python3-virtualenv
```

Create the virtual environment
```
virtualenv venv
```
Activate the virtual environment
```
source venv/bin/activate
```
Install the required libraries listed in `requirements.txt`
```
pip install -r requirements.txt
```
### Test the environment
```
python main.py -m BASIC -d DATASETS/dataset1_PREPROCESSED/ -i 50x1 -e 3
```
Check the input parameters with
```
 python main.py --help
```
### Analyzing the Results
The performances results are printed on file in the `logs` folder, 
Moreover, tensorboard data is stored in the folder `tensorboard` and can be used to print graphs on the training 
loss/acc trend by running
```
tensorboard --logdir tensorboard/
```
and then visualize the graphs at `http://localhost:6006/`.