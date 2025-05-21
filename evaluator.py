"""
Run recommendation/data evaluation and saves outputs

Usage:
    evaluator.py [options] --popular
    evaluator.py [options] --itemknn
    evaluator.py [options] --implicitmf
    evaluator.py [options] --bpr
    evaluator.py [options] --data

Options:
    -v, --verbose       enable verbose log output
    --log-file=FILE     write log file to FILE

Modes:
    --popular              run popScorer evaluation
    --itemknn              run implicit itemKNN evaluation
    --implicitmf           run implicitMF evaluation
    --bpr                  run bpr evaluation  
    --data                 run data evaluation
"""

import logging
import sys

from docopt import ParsedOptions, docopt
from src.scripts import evaluation, data_evaluation
from lenskit.logging import LoggingConfig, get_logger

_log = get_logger('evaluator')

def main(args: ParsedOptions):

    args = docopt(__doc__)
    lcfg = LoggingConfig()

    if args["--verbose"]:
        lcfg.set_verbose()  
    if args["--log-file"]:
        lcfg.log_file(args["--log-file"], logging.DEBUG) 

    lcfg.apply()

    if args["--popular"]:
        _log.info("Running PopScorer evaluation")
        evaluation.evaluaterecs('popular')
    elif args["--itemknn"]:
        _log.info("Running ItemKNN evaluation")
        evaluation.evaluaterecs('itemknn')
    elif args["--implicitmf"]:
        _log.info("Running ImplicitMF evaluation")
        evaluation.evaluaterecs('implicitmf')    
    elif args["--bpr"]:
        _log.info("Running bpr evaluation")
        evaluation.evaluaterecs('bpr') 
    elif args["--data"]:
        _log.info("Running data evaluation")
        data_evaluation.main()        
    else:
        _log.error("Error: the given algorithm is not defined!")
        sys.exit(2)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
