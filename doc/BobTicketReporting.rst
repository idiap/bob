.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Wed Jan 11 14:43:35 2012 +0100
.. 
.. Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

===================================================
Guidelines for Bug Reporting and Feature Requesting
===================================================

Reporting bugs that will be taken into account and treated quickly is not an
easy task. Please read this entire document before submitting tickets. *It
will help us fix your bug faster!* Bug submissions that do not meet the
following guidelines tend to be investigated last or, more likely, closed as
invalid. (For great insights on bug reporting, also read through the very good
suggestions offered by `Eric Raymond's How To Ask Questions The Smart Way`_ and
`Simon Tatham's How to Report Bugs Effectively`_.)

Guidelines
----------

Useful bug reports get bugs fixed, and normally have the following three
qualities:

1. **Reproducible**: If a developer can't see a bug or conclusively prove that
   it exists, the developer will probably not be able to fix it. He or she will
   move on to the next bug. Every relevant detail you provide helps.
2. **Specific**: Give us details. If you're crashing on a certain action,
   please take the time to isolate what is triggering the crash, and include it
   in the bug report if possible. The quicker the developer can isolate the
   issue to a specific problem, the more likely it will be fixed in a timely
   manner.
3. **Not a Duplicate**: Please `search`_ before filing bug reports. Try a
   variety of words in case the one(s) you're using isn't the usual one to
   describe what you're talking about. If you're running a nigthly release look
   at the associated milestone on the  Roadmap to see if your bug is already
   there. Time spent by the team sorting through different tickets on duplicate
   issues may be time not spent fixing a bug.

If you don't find an existing ticket about your issue, then you get to file
one! If you do find an existing ticket about your problem, then it has already
been reported. This is a good thing. If you wish, you can vote on the ticket by
clicking the "up" arrow at the top of the page.

Once you have filled a bug or a feature request, try following it up (mails are
sent when they get changed) and answering any questions that have been asked,
following all steps that may have been suggested. If possible, include
additional useful information.

Examples of good and bad reports
--------------------------------

Let's say you crash while running a |project| program, and want to write up a
bug report:

* **Bad report**: "My program crashed. I think I am not sure what it was. I
  think that this is a really bad problem and you should fix it or else nobody
  will use your library.  By the way, my sister thinks your web pages really
  suck. Oh, that is also the opinion of other folks down here. Thx 4 fixing
  theze bugz."
* **Useful report**: "|project| crashes immediately each time I try to use the
  bob::LBP4R type, using version 2.1.4 on a Mac OS X 10.5.4 system.  It
  crashes upon the initialization of the class. I am attaching the crash
  reports here, together with a small program that I wrote that can reproduce
  the issue."

Composing a new ticket
----------------------

Once you have logged into your Trac account you can click `New Ticket`_ at the
top of the page to start creating a new ticket.

A few things to note:

* Please only put one issue in each ticket. File multiple tickets if you can
  confirm multiple bugs or want multiple features implemented.
* Please make sure to file your ticket in English, as that is the only language
  that most of our developers understand.
* Please keep the comments related to fixing the bug or relevant to the feature
  request. Comments like "Me too" or "I really want this fixed" are not usually
  helpful.
* Please include the following information in every bug report you file:

  - A list of steps to reproduce the problem. If you need to compile a program
    to cause the bug, please create a small version of the program that still
    causes the crash and upload it with the bug report. Do not upload files
    that only make sense together with other files you are not uploading -
    nobody will look at these. Focus in finding an easy way to explain your
    problem and make it reproducible;
  - The specific versions of |project| and the operating system that you are
    using (*not* "I'm using the latest/previous version"). You can obtain
    version numbers by executing the following script that is provided with
    |project|:

.. code-block:: sh
   
   $ info_table.py

.. Place here all external references

.. _Eric Raymond's How To Ask Questions The Smart Way: http://www.catb.org/~esr/faqs/smart-questions.html
.. _Simon Tatham's How to Report Bugs Effectively: http://www.chiark.greenend.org.uk/~sgtatham/bugs.html
.. _search: https://github.com/idiap/bob/issues
.. _new ticket: https://github.com/idiap/bob/issues
