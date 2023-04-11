import glob
import logging
import os

from RAPIDprep import is_valid_rapid_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='check.log',
    filemode='w'
)

for d in glob.glob(os.path.join('/tdxrapid', '*')):
    is_valid_rapid_dir(d)
