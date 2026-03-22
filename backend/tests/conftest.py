import sys
import os

# Add backend/ to sys.path so bare imports (from search_tools import ...) work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
