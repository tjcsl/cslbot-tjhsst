# -*- coding: utf-8 -*-
# Copyright (C) 2013-2018 Tris Wilson, Peter Foley, Srijay Kasturi, Samuel Damashek, James Forcier and Reed Koser
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

_RNN_DIR = '/home/peter/char-rnn-tensorflow'
_CHECKPOINT_PATTERN = 'save/model.ckpt-%d.index'


@Command('brain', ['nick'], limit=5)
def cmd(send, msg, args):
    """Neural networks!

    Syntax: !brain

    """
    # FIXME: this whole thing is a god-awful hack
    latest = 0
    for f in os.scandir(os.path.join(_RNN_DIR, 'save')):
        match = re.match(r'model.ckpt-(\d+).index', f.name)
        if match is None:
            continue
        latest = max(latest, int(match.group(1)))
    send("Sampling output at checkpoint %d" % latest)
    output = subprocess.check_output([os.path.join(_RNN_DIR, 'sample.py')], cwd=_RNN_DIR, universal_newlines=True)
    for line in output.splitlines():
        send(line, target=args['nick'])
