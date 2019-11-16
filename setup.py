#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2013-2016 Samuel Damashek, Peter Foley, James Forcier, Srijay Kasturi, Reed Koser, Christopher Reffett, and Fox Wilson
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from setuptools import find_packages, setup

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(name="cslbot-tjhsst",
      description="TJHSST-specific commands for CslBot.",
      long_description=long_description,
      author="The TJHSST Computer Systems Lab",
      author_email="cslbot@pefoley.com",
      url="https://github.com/tjcsl/cslbot-tjhsst",
      version="0.4",
      license="GPL",
      zip_safe=False,
      packages=find_packages(),
      setup_requires=['setuptools_git'],
      install_requires=['CslBot', 'feedparser'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Communications :: Chat :: Internet Relay Chat',
      ])
