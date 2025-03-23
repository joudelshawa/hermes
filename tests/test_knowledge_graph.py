import sys
import os
print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents import Hermes
import os
import json

if __name__ == "__main__":
    print("yes!")