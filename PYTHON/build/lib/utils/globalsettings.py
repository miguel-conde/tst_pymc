import os
import configparser

class prjSettings():
    def __init__(self):
        pass

the_folders = prjSettings()

the_folders.DIR_ROOT  = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..") # Project root is defined by globalsettings.py location
the_folders.DIR_DATA = os.path.join(the_folders.DIR_ROOT, "data")
the_folders.DIR_DATA_RAW = os.path.join(the_folders.DIR_DATA, "raw")
the_folders.DIR_DATA_CLEAN = os.path.join(the_folders.DIR_DATA, "clean")

the_files = prjSettings()

the_files.CFG_FILE = os.path.join(the_folders.DIR_ROOT, "config.ini")

## CONSTANTS

the_constants = prjSettings()

## CONFIG FILE
prj_cfg = prjSettings()

config = configparser.ConfigParser()

config.read(the_files.CFG_FILE)

prj_cfg.default_model               = config['DEFAULT']['model']
prj_cfg.default_directiva           = config['DEFAULT']['directiva_del_sistema']
prj_cfg.default_max_tokens_contexto = config['DEFAULT']['max_tokens_contexto']
prj_cfg.default_max_tokens          = config['DEFAULT']['max_tokens']