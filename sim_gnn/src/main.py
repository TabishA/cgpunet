"""SimGNN runner."""

from utils import tab_printer, process_pair
from simgnn import SimGNNTrainer
from param_parser import parameter_parser
import argparse

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
    trainer.score()
    if args.save_path:
        trainer.save()

if __name__ == "__main__":
    args = parameter_parser()

    tab_printer(args)
    trainer = SimGNNTrainer(args)

    data = process_pair('../dataset/test/119.json')

    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
    # trainer.score()
    prediction = trainer.predict(data)
    print(prediction)
    if args.save_path:
        trainer.save()
