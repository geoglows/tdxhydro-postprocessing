import glob
import logging
import os

from RAPIDprep import validate_rapid_directory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='check.log',
    filemode='w'
)

for d in glob.glob(os.path.join('/tdxrapid', '*')):
    validate_rapid_directory(d)
