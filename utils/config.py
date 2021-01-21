import time
import os

# Timestamp Execution
timeExec = "{}".format(time.strftime("%d%m_%H%M"))

# Getting the Absolute Path to the main.py script,
# NB! Asssuming that the name of this script is 'main.py' (7 letters)
MAIN_PATH = os.path.realpath("main.py")[:-7]