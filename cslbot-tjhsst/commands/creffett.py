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

from cslbot.helpers import misc, textutils
from cslbot.helpers.command import Command


@Command(['creffett', 'rage'], ['nick', 'target', 'db', 'do_kick', 'botnick', 'name'])
def cmd(send, msg, args):
    """RAGE!!!

    Syntax: !rage <text>

    """
    if args['name'] == 'creffett':
        if not args['nick'].startswith('creffett') and args['nick'] != args['botnick']:
            send("You're not creffett!")
            send(misc.ignore(args['db'], args['nick']))
            if args['target'] != 'private':
                args['do_kick'](args['target'], args['nick'], 'creffett impersonation')
            return
    if not msg:
        send("Rage about what?")
        return
    # c.send_raw("MODE %s -c" % CHANNEL)
    send(textutils.gen_creffett(msg))
    # c.send_raw("MODE %s +c" % CHANNEL)
    send('</rage>')
