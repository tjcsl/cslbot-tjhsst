#!/usr/bin/env python3
# Copyright (C) 2013-2015 Samuel Damashek, Peter Foley, James Forcier, Srijay Kasturi, Reed Koser, Christopher Reffett, and Fox Wilson
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

from setuptools import setup, find_packages

setup(
    name="cslbot-tjhsst",
    description="TJHSST-specific commands for CslBot.",
    author="The TJHSST Computer Systems Lab",
    author_email="cslbot@pefoley.com",
    url="https://github.com/tjcsl/cslbot-tjhsst",
    version="0.1",
    license="GPL",
    zip_safe=False,
    packages=find_packages(),
    install_requires=['CslBot'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Programming Language :: Python :: 3.4',
        'Topic :: Communications :: Chat :: Internet Relay Chat',
        ],
)
