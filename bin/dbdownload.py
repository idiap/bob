#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jul 25 20:50:52 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Downloads databases distributed with bob.

This program will download named databases from Bob's Idiap server. It will
first check if the files are not up-to-date and, if that is not the case,
download them. 

This program is an extension of the program "OpenAnything": a kind and
thoughtful library for HTTP web services. Which is part of 'Dive Into Python',
a free Python book for experienced programmers.  Visit
http://diveintopython.org/ for the latest version.
"""

USER_AGENT = 'OpenAnything/1.6 +http://diveintopython.org/http_web_services/'
SERVER = 'http://www.idiap.ch/software/bob/databases'
VERSION = 'nightlies/last'
DESTINATION = 'databases'

__epilog__ = """Example usage:

1. Downloads 2 database from the standard server to the '%(destination)s'
   folder (at the current directory):

  $ %%(prog)s banca replay

2. Activate verbosity and see what's going on:

  $ %%(prog)s --verbose banca

3. Downloads a version different than "%(version)s" (the default):

  $ %%(prog)s --version='releases/1.0.2' replay

4. Downloads from a different server than "%(server)s" (the default). Note we
   will append the version string (as indicated above) to the final destination
   URL.

  $ %%(prog)s --server='http://www.example.com/bob' multipie
""" % {'server': SERVER, 'version': VERSION, 'destination': DESTINATION}

import os
import sys
import argparse
import urllib2, urlparse, gzip
from StringIO import StringIO

class SmartRedirectHandler(urllib2.HTTPRedirectHandler):
  def http_error_301(self, req, fp, code, msg, headers):
    result = urllib2.HTTPRedirectHandler.http_error_301(
      self, req, fp, code, msg, headers)
    result.status = code
    return result

  def http_error_302(self, req, fp, code, msg, headers):
    result = urllib2.HTTPRedirectHandler.http_error_302(
      self, req, fp, code, msg, headers)
    result.status = code
    return result

class DefaultErrorHandler(urllib2.HTTPDefaultErrorHandler):
  def http_error_default(self, req, fp, code, msg, headers):
    result = urllib2.HTTPError(
      req.get_full_url(), code, msg, headers, fp)
    result.status = code
    return result

def openAnything(source, etag=None, lastmodified=None, agent=USER_AGENT):
  """URL, filename, or string --> stream

  This function lets you define parsers that take any input source
  (URL, pathname to local or network file, or actual data as a string)
  and deal with it in a uniform manner.  Returned object is guaranteed
  to have all the basic stdio read methods (read, readline, readlines).
  Just .close() the object when you're done with it.

  If the etag argument is supplied, it will be used as the value of an
  If-None-Match request header.

  If the lastmodified argument is supplied, it must be a formatted
  date/time string in GMT (as returned in the Last-Modified header of
  a previous request).  The formatted date/time will be used
  as the value of an If-Modified-Since request header.

  If the agent argument is supplied, it will be used as the value of a
  User-Agent request header.
  """

  if hasattr(source, 'read'):
    return source

  if source == '-':
    return sys.stdin

  if urlparse.urlparse(source)[0] == 'http':
    # open URL with urllib2
    request = urllib2.Request(source)
    request.add_header('User-Agent', agent)
    if lastmodified:
      request.add_header('If-Modified-Since', lastmodified)
    if etag:
      request.add_header('If-None-Match', etag)
    request.add_header('Accept-encoding', 'gzip')
    opener = urllib2.build_opener(SmartRedirectHandler(), DefaultErrorHandler())
    return opener.open(request)

  # try to open with native open function (if source is a filename)
  try:
    return open(source)
  except (IOError, OSError):
    pass

  # treat source as string
  return StringIO(str(source))

def fetch(source, etag=None, lastmodified=None, agent=USER_AGENT):
  '''Fetch data and metadata from a URL, file, stream, or string'''
  result = {}
  result['gzip'] = False #becomes True if data is transmitted in gzip format
  f = openAnything(source, etag, lastmodified, agent)
  result['data'] = f.read()
  if hasattr(f, 'headers'):
    # save ETag, if the server sent one
    result['etag'] = f.headers.get('ETag')
    # save Last-Modified header, if the server sent one
    result['lastmodified'] = f.headers.get('Last-Modified')
    if f.headers.get('content-encoding') == 'gzip':
      # data came back gzip-compressed, decompress it
      result['gzip'] = True
      result['data'] = gzip.GzipFile(fileobj=StringIO(result['data'])).read()
  if hasattr(f, 'url'):
    result['url'] = f.url
    result['status'] = 200
  if hasattr(f, 'status'):
    result['status'] = f.status
  f.close()
  return result

def download(dbname, server, version, destdir, verbose):

  url = '/'.join((server.strip('/'), version.strip('/'), dbname + '.sql3'))

  # If the destdir does not exist, create it
  if not os.path.exists(destdir): 
    if verbose: print "Creating directory '%s'..." % destdir
    os.makedirs(destdir)

	# Destination is the name of the directory plus the name of the database
  destination = os.path.join(destdir, dbname + '.sql3')

  # If destination exists and it has a .etag file with it, read it and
  # give it to the fetch method - this will avoid re-downloading databases
  # that are up-to-date.
  etag = None
  if os.path.exists(destination + '.etag'):
    etag = open(destination + '.etag', 'rt').read().strip()

  if verbose:
    print "Requesting %s" % (url,)

  # Fetch the data, if not already up-to-date
  data = fetch(url, etag=etag)

  if data['status'] == 200:
    output = open(destination, 'wb')
    output.write(data['data'])
    output.close()

    if verbose:
      print "Gzip Compression: %s" % data['gzip']
      print "Database Size: %d bytes" % len(data['data'])
      print "Last Modification: %s" % data['lastmodified']
      print "Saved at: %s" % destination

    if data['etag']:
      if verbose:
        print "E-Tag: %s" % data['etag']
      etag_file = open(destination + '.etag', 'wt')
      etag_file.write(data['etag'])
      etag_file.close()
      print "E-Tag cached: %s" % (destination + '.etag',)

  elif data['status'] == 304: #etag matches
    if verbose:
      print "Currently installed version is up-to-date (did not re-download)"

  else:
    raise IOError, "Failed download of %s (status: %d)" % (url, data['status'])

def main():
  """Main function: parses options and download all databases available on
     the server."""
  
  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("-s" , "--server", metavar='URL', type=str, 
      default=SERVER,
      help="the name of the server to download from (defaults to %(default)s)")
  parser.add_argument("-r" , "--version", metavar='PATH', type=str,
      default=VERSION, help="the version to download (defaults to %(default)s)")
  parser.add_argument("-d" , "--destination", metavar='DIR', type=str,
      default=DESTINATION, 
      help="where to download the databases (defaults to %(default)s)")
  parser.add_argument("-v", "--verbose", dest="verbose", default=False,
      action='store_true', help="enable verbose output")
  parser.add_argument("database", metavar='DATABASE', type=str, nargs='+',
      help="names of databases to download")

  args = parser.parse_args()

  for db in args.database: 
    download(db, args.server, args.version, args.destination, args.verbose)

if __name__ == '__main__':
  main()
