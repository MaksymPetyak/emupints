#
# Root of the emupints module.
# Provides access to different emulator classes
#
# This file is part of emupints.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys

#
# Version info: Remember to keep this in sync with setup.py!
#
VERSION_INT = 0, 1, 1
VERSION = '.'.join([str(x) for x in VERSION_INT])
if sys.version_info[0] < 3:
    del(x)  # Before Python3, list comprehension iterators leaked


#
# Expose pints version
#
def version(formatted=False):
    if formatted:
        return 'emupints ' + VERSION
    else:
        return VERSION_INT


#
# Different types of emulators
#
from ._emulator import Emulator
from ._gp_emulator import GPEmulator


__all__ = ["Emulator", "GPEmulator"]


#
# Remove any imported modules
#
del(sys)
