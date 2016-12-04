# -*- coding: utf-8 -*-
# Copyright (C) 2013-2016 Fox Wilson, Peter Foley, Srijay Kasturi, Samuel Damashek, James Forcier and Reed Koser
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

import re
import os
import subprocess

from cslbot.helpers.command import Command

_RNN_DIR = '/home/peter/torch-rnn'
_TH_PATH = '/home/peter/torch/install/bin/th'
_CHECKPOINT_PATTERN = 'cv/checkpoint_%d.t7'


@Command('brain', ['nick'], limit=5)
def cmd(send, msg, args):
    """Neural networks!

    Syntax: !brain (latest)

    """
    # FIXME: this whole thing is a god-awful hack
    latest = 0
    for f in os.scandir(os.path.join(_RNN_DIR, 'cv')):
        match = re.match('checkpoint_(\d+).t7', f.name)
        if match is None:
            continue
        latest = max(latest, int(match.group(1)))
    if msg == "latest":
        send(latest)
        return
    output = subprocess.check_output([_TH_PATH, 'sample.lua', '-checkpoint', _CHECKPOINT_PATTERN % latest, '-gpu', '-1'], cwd=_RNN_DIR)
    lines = [line.strip() for line in output.decode('ascii', 'ignore').splitlines()]
    for line in lines:
        if line:
            send(line, target=args['nick'])
