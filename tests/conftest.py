import os
import sys

# Make src/ importable as `import pipeline` from any test file, without
# repeating this path hack at the top of every test module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))