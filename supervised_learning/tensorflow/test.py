import os
print("Started")
import tensorflow as tf
from imp import load_source
print("Importing modules...")
try:
    train_module = load_source('train_module', '6-train.py')
    print("Train module imported.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Crashed during import!")
