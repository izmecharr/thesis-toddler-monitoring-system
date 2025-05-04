"""
This is a PyInstaller hook file to help package PyQt5 applications correctly.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all submodules
hiddenimports = collect_submodules('PyQt5')

# Collect necessary data files
datas = collect_data_files('PyQt5')