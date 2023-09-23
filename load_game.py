__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import tkinter as tk
from tkinter import filedialog
import sys
from cleaners import CleanersGame

def main(argv):
    # Load the defaults
    from settings import game_settings


    if game_settings['visSpeed'] != 'normal' and game_settings['visSpeed'] != 'fast' and game_settings['visSpeed'] != 'slow':
        print("Error! Invalid setting '%s' for visualisation speed.  Valid choices are 'slow','normal',fast'" % game_settings['visSpeed'])
        sys.exit(-1)


    # If load game wasn't specified in the command line arguments then
    # open a dialog box in the 'saved' folder
    root = tk.Tk()
    root.withdraw()

    loadGame = filedialog.askopenfilename(initialdir="./saved")

    # Load a previously saved game
    CleanersGame.load(loadGame,visResolution=game_settings['visResolution'],
               visSpeed=game_settings['visSpeed'])


if __name__ == "__main__":
   main(sys.argv[1:])