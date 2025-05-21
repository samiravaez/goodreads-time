"""
Run recommendation models and saves outputs

Usage:
    recommender.py [options] --popular
    recommender.py [options] --itemknn
    recommender.py [options] --implicitmf
    recommender.py [options] --bpr

Options:
    -v, --verbose       enable verbose log output
    --log-file=FILE     write log file to FILE

Modes:
    --popular              run popScorer recommender
    --itemknn              run implicit itemKNN recommender
    --implicitmf           run implicitMF recommender
    --bpr                  run BPR recommender
"""

import logging
import sys

from docopt import ParsedOptions, docopt
from sandal.cli import setup_logging
from src.scripts import generaterecs
from lenskit.logging import LoggingConfig, get_logger

_log = get_logger('recommender')

def main(args: ParsedOptions):

    args = docopt(__doc__)
    lcfg = LoggingConfig()

    if args["--verbose"]:
        lcfg.set_verbose()  
    if args["--log-file"]:
        lcfg.log_file(args["--log-file"], logging.DEBUG) 

    lcfg.apply()

    if args["--popular"]:
        _log.info("Running PopScorer")
        generaterecs.runmodel('popular')
    elif args["--itemknn"]:
        _log.info("Running ItemKNN")
        generaterecs.runmodel('itemknn')
    elif args["--implicitmf"]:
        _log.info("Running ImplicitMF")
        generaterecs.runmodel('implicitmf')    
    elif args["--bpr"]:
        _log.info("Running BPR")
        generaterecs.runmodel('bpr')    
    else:
        _log.error("Error: the given algorithm is not defined!")
        sys.exit(2)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
