# Copyright (C) 2013-2014 Fox Wilson, Peter Foley, Srijay Kasturi, Samuel Damashek, James Forcier and Reed Koser
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

from cslbot.helpers.command import Command
from cslbot.helpers import arguments, textutils


@Command(['fwilson', 'son'], ['config'])
def cmd(send, msg, args):
    """Imitates fwilson.
    Syntax: !fwilson (-f|w) <message>
    """
    parser = arguments.ArgParser(args['config'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', action='store_true')
    group.add_argument('-w', action='store_true')
    parser.add_argument('msg', nargs='*')
    try:
        cmdargs = parser.parse_args(msg)
    except arguments.ArgumentException as e:
        send(str(e))
        return
    msg = " ".join(cmdargs.msg) if cmdargs.msg else textutils.gen_word()
    mode = None
    if cmdargs.f:
        mode = 'f'
    elif cmdargs.w:
        mode = 'w'
    send(textutils.gen_fwilson(msg, mode))
