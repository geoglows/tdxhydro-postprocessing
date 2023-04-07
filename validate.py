import glob
import logging
import os

from RAPIDprep import validate_rapid_directory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='validate.log',
    filemode='w'
)

for d in glob.glob(os.path.join('/tdxprocessed', '*')):
    validate_rapid_directory(d)
