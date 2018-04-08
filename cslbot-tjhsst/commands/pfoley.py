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

import random

from cslbot.helpers.command import Command

import feedparser
import random

ARS_FEED_URL = "https://feeds.feedburner.com/arstechnica/index"


@Command(['pfoley'])
def cmd(send, *_):
    """Imitates pfoley.

    Syntax: !pfoley

    """
    if random.random() < 0.5:
        send("ok, and?")
    else:
        feed = feedparser.parse(ARS_FEED_URL)
        entry = random.choice(feed['entries'])
        send('{} - {}'.format(entry['link'], entry['title']))
