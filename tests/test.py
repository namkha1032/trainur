from trainur import Trainur
from logab import log_wrap
from fire import Fire
from dataclasses import dataclass

class CustomTrainer(Trainur):
    train_var:any=1
    def train(self):
        pass
    
    def inference(self):
        pass

class CustomTester(CustomTrainer):
    test_var:any="some str"
    def test(self):
        pass

if __name__ == "__main__":
    with log_wrap():
        Fire(CustomTester)