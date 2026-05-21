import pycodestyle
import sys
import glob

files = glob.glob('c:/Users/USER/Desktop/alu-machine_learning/supervised_learning/tensorflow/*.py')
style = pycodestyle.StyleGuide()
result = style.check_files(files)
sys.exit(result.total_errors)
