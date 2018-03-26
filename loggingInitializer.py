# coding=utf-8
import logging
import sys

file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    #format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    format = '[{%(lineno)d}  - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('LOGGER_NAME')