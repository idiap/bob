#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Wed 05 Aug 11:36 2015 CEST
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

import os
import pkgtools.pypi
import distutils.version
import pkg_resources

def get_config():
  """
  Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__)


def get_url(package_name):
  "Given a package name get, from PyPI, the URL name"
  import pkgtools.pypi
  return pkgtools.pypi.PyPIJson(package_name).retrieve()['urls'][0]['url']


def download(url, output_dir="."):
  """
  Download a file given the URL
  
  **Parameters**:
  
    url: The URL
    output_dir: The directory that stores the file
  
  """

  import six
  import os
  
  file_name = url.split('/')[-1] #Getting only the file name without the version
  file_name = os.path.join(output_dir,file_name)
  u = six.moves.urllib.request.urlopen(url)
  f = open(file_name, 'wb')
  meta = u.info()
  file_size = int(meta.get("Content-Length")[0])
  print ("Downloading: %s Bytes: %s" % (file_name, file_size))

  file_size_dl = 0
  block_sz = 8192
  while True:
    buffer = u.read(block_sz)
    if not buffer:
      break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
    status = status + chr(8)*(len(status)+1)
    print (status)

  f.close()
  return file_name



def download_packages(requirements,  output_dir="./temp"):
  """
  This function downloads and unpacks all the required packages to a temp directory,
  so this can be used in the future to build an integrated documentation
  
  **Parameters**:

  requirements: The list of files to be downloaded
  
   
  """ 
  import zipfile
  import os

  import bob.io.base    
  bob.io.base.create_directories_safe(output_dir)

  urls = []
  print ("Fetching urls!!!")
  for r in requirements:
    try:
      package_name = r.split("=")[0]
      print ("  Fetching {0}".format(package_name))
      urls.append(get_url(package_name))
    except HTTPError as exc:
      # url request failed with a something else than 404 Error
      print ("Requesting URL %s returned error %s" % (url, exc))
  
  for u in urls:
    file_name = download(u, output_dir)

    print ("Unziping {0}".format(u)    )
    f = open(file_name, 'rb')
    z = zipfile.ZipFile(f)
    z.extractall(os.path.dirname(file_name))
    f.close()
    os.rename(file_name.rstrip(".zip"),file_name.split("-")[0])
    os.unlink(file_name)


def get_releases(package):
  """
  Given a package name, get the release versions
  """  
  try:
    return pkgtools.pypi.PyPIJson(package).retrieve()['releases'].keys()
  except:
    return []


def get_max_version(versions):

  try:
    v = list(reversed(sorted([distutils.version.StrictVersion(k) for k in versions])))
    final = [k for k in v if not k.prerelease]
    if final: return final[0]
    return v[0]
  except:
    v = list(reversed(sorted([distutils.version.LooseVersion(k) for k in versions])))
    final = [k for k in v if not re.search(r'[a-z]', k.vstring)]
    if final: return final[0]
    return v[0]
    

def get_dependencies(pkg_name="bob"):
  """
  Given a package name, get the dependency list
  """  
  package      = pkg_resources.working_set.by_key[pkg_name]
  return [str(r) for r in package.requires()]

  
