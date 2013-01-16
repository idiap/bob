# Original authors: Anders Chrigstrom, Ronny Wikh
# author e-mail: Ronny.Wikh@gmail.com
# source: http://code.google.com/p/insulatenoseplugin/
# license: LGPL-2.1

#
# Test insulation module. 
#
# TODO: prints in tests (captured stuff)
# Copyright (c) 2008 Sungard Front Arena

import sys, pickle, os, types, traceback
import socket, select
from nose.plugins import Plugin
from nose.plugins.skip import SkipTest
from nose.result import TextTestResult
from cStringIO import StringIO

from subprocess import Popen, PIPE

# Python2.3 doesn't have a devnull in os, fudge some reasonable fallbacks here
devnull = getattr(os, 'devnull', None)
if devnull is None:
    if os.name == 'nt':
        devnull = 'nul'
    elif os.name == 'posix':
        devnull = '/dev/null'

"""
Return the name of a test method.

Arguments: test - the test to find the name of
Returns:   The test name
"""

if sys.version_info >= (2, 5):
    def getMethodName(test):
        return test._testMethodName
else:
    def getMethodName(test):
        return test._TestCase__testMethodName

class Insulate(Plugin):
    "Master insulation plugin class"

    score = sys.maxint
    name = 'insulate'
    restart_after_crash = True
    show_slave_output = False
    enableOpt = 'insulate'
    testCount = 0
    testSlave = None
    
    def options(self, parser, env=os.environ):
        super(Insulate, self).options(parser, env)

        parser.add_option('--insulate-skip-after-crash',
                          dest='restart_after_crash',
                          action="store_false",
                          default=True)
        parser.add_option('--insulate-not-in-slave',
                          dest='not_in_slave',
                          type='string',
                          action="append",
                          default=[])
        parser.add_option('--insulate-in-slave',
                          dest='in_slave',
                          type='string',
                          action="append",
                          default=[])
        parser.add_option('--insulate-show-slave-output',
                          dest='show_slave_output',
                          action="store_true",
                          default=False)

    def configure(self, options, conf):
        super(Insulate, self).configure(options, conf)
        if getattr(options, 'insulateslave', None):
            # Insulate plugin should never be enabled in slave
            self.enabled = False
        if self.enabled:
            self.restart_after_crash = options.restart_after_crash
            self.not_in_slave = options.not_in_slave[:] 
            self.in_slave = options.in_slave[:]

            self.show_slave_output = options.show_slave_output

            # Always enabled in slave
            self.in_slave.append('--nocapture')

        self.argv = [sys.executable] + sys.argv		# Save for slave creation in prepareTest

    def prepareTest(self, test):
        """
        Prepare a test run; set counter to zero in preparation for run,
        don't mess with the test itself in this case.

        Arguments: test - test to prepare
        Returns:   None
        """
        self.testCount = 0
        args = [arg for arg in self.argv if arg not in self.not_in_slave]
        args.extend(self.in_slave)
        self.testSlave = TestSlave(self.restart_after_crash,
                                   self.show_slave_output,
                                   args)

    def finalize(self, result):
        """
        Finalise the test run; eliminate the 'spent' slave so that a new
        can be created at next test run

        Arguments: result
        Returns:   None
        """
        self.testSlave.dropSlave()
        self.testSlave = None

    def prepareTestCase(self, test):
        """
        Prepare a specific test; wrap it for processing, and return
        the wrapper.

        Arguments: test - the test to wrap
        Returns:   Wrapped test
        """
        return TestWrapper(self, test.test)

    def runTestInSlave(self, test):
        """
        Run a test in a slave object.

        Arguments: test - test to run
        Returns: result status
        """
        self.testCount += 1

        result = self.testSlave.runTest(self.testCount, test)
        return result

class TestWrapper(object):
    "This class is a wrapper for running tests in a separate process."

    def __init__(self, plugin, orgtest):
        """
        Initialise the object

        Arguments: plugin - the insulate plugin running the test
                   orgtest - the unwrapped test
        Returns:   None
        """
        self.plugin = plugin
        self.orgtest = orgtest

    if sys.version_info >= (2,5):
        def _exc_info(self):
            return self.orgtest._exc_info()
    else:
        def _exc_info(self):
            return self.orgtest._TestCase__exc_info()

    def __call__(self, result):
        """
        Runs the test in the test slave, retrieving result data and
        putting it in the supplied result object.

        This actually is just an adapatation of the
        'unittest.TestCase.run' method.

        Arguments: result - where to put the results
        Returns:   None
        """
        if result is None:
            result = self.orgtest.defaultTestResult()

        result.startTest(self.orgtest)
        method_name = getMethodName(self.orgtest)
        testMethod = getattr(self.orgtest, method_name)

        try:
            status, (ev, tb), data = self.plugin.runTestInSlave(self.orgtest)
            if isinstance(ev, types.InstanceType):
                et = ev.__class__
            else:
                et = type(ev)
                
            exc_info = et, ev, tb
            if status == ResultCollector.SUCCESS:
                result.addSuccess(self.orgtest)

            elif status == ResultCollector.FAILURE:
                result.addFailure(self.orgtest, exc_info)

            elif status == ResultCollector.ERROR:
                result.addError(self.orgtest, exc_info)

            elif status == ResultCollector.ABORT:
                result.addError(self.orgtest, exc_info)

            elif status == ResultCollector.SKIP:
                result.addSkip(self.orgtest, data['reason'])

            else:
                raise RuntimeError('Protocol error in master/slave communications')

        finally:
            result.stopTest(self.orgtest)

# Picklable objects mimicking a traceback structure enough
class Code(object):
    def __init__(self, code):
        self.co_filename = code.co_filename
        self.co_name = code.co_name

class Frame(object):
    def __init__(self, frame):
        self.f_globals = {}
        if '__unittest' in frame.f_globals:
            self.f_globals['__unittest'] = 1
        self.f_code = Code(frame.f_code)

class Traceback(object):
    def __init__(self, tb):
        self.tb_lineno = tb.tb_lineno
        self.tb_frame = Frame(tb.tb_frame)
        self.tb_next = self.make(tb.tb_next)

    def make(cls, tb):
        if tb is None:
            return None
        return cls(tb)
    make = classmethod(make)

class TestSlave(object):
    "The test slave class for running tests in a separate process."

    def __init__(self, restart_after_crash, show_slave_output, args):
        """
        Initialise the object, starting a 'slave' nosetest (or whatever
        the program was called) object in a separate process.

        Arguments: restart_after_crash - restart the slave after a crash
                   show_slave_output - show slave stdout, err
                   args - argument list
        Returns:   None
        """
        self.args = args

        self.restart_after_crash = restart_after_crash
        self.show_slave_output = show_slave_output
        self.fromSlave = None
        self.toSlave = None
        self.noseSlave = None
        self.hasCrashed = False
        self.stdouterr = []

    def startSlave(self):
        """
        Creates a slave process.

        Arguments: None
        Returns:   None
        """
        if self.noseSlave is not None:
            return

        if self.hasCrashed and not self.restart_after_crash:
            return

        # Create a listen socket bound to localhost and any port.
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        port = sock.getsockname()[1] # Get the port number we got
        args = self.args[:]
        args.append('--with-insulateslave=%d'%(port))
        sin = open(devnull, 'r')
        if self.show_slave_output:
            noseSlave = Popen(args, stdin=sin)
        else:
            sout = open(devnull, 'w')
            noseSlave = Popen(args, stdin=sin, stdout=sout, stderr=sout)

        iwtd = []
        while len(iwtd) == 0:
            iwtd, _, _ = select.select([sock], [], [], 2.0)

            if noseSlave.poll() is not None:
                sock.close()
                raise CrashInTestError()
        slaveSock, _ = sock.accept()
        sock.close()

        slavefile = slaveSock.makefile('rwb')
        self.noseSlave = noseSlave
        self.toSlave = slavefile
        self.fromSlave = slavefile

    def dropSlave(self):
        """
        Drop a slave after crash

        Arguments: None
        Returns:   None
        """
        if self.noseSlave is not None:
            try:
                self.sendToSlave(None) # Tell the slave we are finished with it.
            except (EOFError, IOError, socket.error), e:
                pass
            self.noseSlave = None
            self.toSlave.close()
            self.fromSlave.close()
            self.toSlave = self.fromSlave = None
        self.hasCrashed = True


    def sendToSlave(self, count):
        pickle.dump(count, self.toSlave, -1)
        self.toSlave.flush()

    def readFromSlave(self):
        return pickle.load(self.fromSlave)

    def runTest(self, count, orgtest):
        """
        Runs the test.

        Argument: count - the test number to run, as counted by both
                          master and slave independently.
        Returns: (result status, result data)
        """
        self.startSlave()

        if self.noseSlave is None:
            return ResultCollector.ERROR, (SkipAfterCrash(' - previous test crashed, skipping.'), None)

        try:
            self.sendToSlave(count)
        except (EOFError, IOError, socket.error), e:
            _, _, tb = sys.exc_info()
            self.dropSlave()
            return ResultCollector.ERROR, (e, tb), None

        try:
            status, exc, (stdout, stderr), data = self.readFromSlave()
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)
            return status, exc, data

        except (EOFError, IOError, socket.error), e:
            _, _, tb = sys.exc_info()
            self.dropSlave()
            return ResultCollector.ERROR, (CrashInTestError(), tb), None

class SkipAfterCrash(SkipTest):
    "After crash marker when using the --insulate-skip-after-crash flag"

class CrashInTestError(Exception):
    "Exception to denote slave death"

class _Infinite(object):
    def __cmp__(self, other):
        return 1 # Always bigger than anything else.
Infinite = _Infinite()

class InsulateSlave(Plugin):
    "Slave insulation plugin class"
    score = sys.maxint
    name = 'insulateslave'
    _nextTest = None

    def options(self, parser, env=os.environ):
        parser.add_option('', '--with-insulateslave',
                          action='store',
                          type="string",
                          dest='insulateslave',
                          default='')

    def configure(self, options, conf):
        """
        """
        if options.insulateslave:
            self.enabled = True
            port = int(options.insulateslave)
            masterSock = socket.socket()
            masterSock.connect(('127.0.0.1', port))
            masterFile = masterSock.makefile('rwb')
            self.fromMaster = masterFile
            self.toMaster = masterFile
            
    def prepareTest(self, test):
        """
        Prepare a specific test; set counter to zero in preparation for run,
        don't mess with the test itself in this case.

        Arguments: None
        Returns:   None
        """
        global testCount
        testCount = 0

    def prepareTestCase(self, test):
        """
        Prepare a specific test; wrap it for processing, and return
        the wrapper.

        Arguments: test - the test to wrap
        Returns:   Wrapped test
        """
        return TestSlaveWrapper(self, test.test)

    def finalize(self, result):
        if self._nextTest is Infinite:
            nextTest = None
        else:
            nextTest = self.getNextTest()
        if nextTest is not None:
            self.sendToMaster(ResultCollector.ABORT, (None, None), ('', ''), None)
        self.toMaster.close()
        self.fromMaster.close()

    def getNextTest(self):
        return pickle.load(self.fromMaster)

    def sendToMaster(self, status, exc_data, io, data, dontRecurse=False):
        """
        Send data to master, but check that we can pickle first

        Arguments: Data to send
        Returns:   None
        """
        try:
            pickledData = pickle.dumps((status, exc_data, io, data), -1)
        except pickle.PicklingError:
            if dontRecurse:
                raise	# Fail horribly, been here already
            exc_info = sys.exc_info()
            self.sendToMaster(ResultCollector.ERROR,
                              (exc_info[1], Traceback.make(exc_info[2])),
                              io, None, True)
            return
        self.toMaster.write(pickledData)
        self.toMaster.flush()

class TestSlaveWrapper(object):
    "This class is a wrapper for running tests in a separate process."

    # Set up a 'nextTest' static variable
    def _getNext(self):
        return self.plugin._nextTest
    def _setNext(self, value):
        self.plugin._nextTest = value
    nextTest = property(_getNext, _setNext)
    del _getNext
    del _setNext

    def __init__(self, plugin, orgtest):
        """
        Initialise the object

        Arguments: orgtest - the unwrapped test
        Returns:   None
        """
        self.plugin = plugin
        self.orgtest = orgtest

    def __call__(self, result):
        """
        Runs the test, pickling data and outputting it to
        the master to retrieve.

        Arguments: result - where to put the results
        Returns:   None
        """
        global testCount

        testCount += 1

        if self.nextTest is None:
            self.nextTest = self.plugin.getNextTest()
            if self.nextTest is None:
                self.nextTest = Infinite
            if self.nextTest < testCount:
                # Should never happen, but should be signalled more clearly
                self.nextTest = None
                self.plugin.sendToMaster(ResultCollector.ABORT, (None, None), ('', ''), None)
                os._exit(9)	# Just die

        if testCount < self.nextTest:
            try:
                raise SkipTest('Skipped test %d'%(testCount,))
            except:
                result.addError(self.orgtest, sys.exc_info())
                return

        self.nextTest = None
        method_name = getMethodName(self.orgtest)
        test = getattr(self.orgtest, method_name)
        res = ResultCollector(result)
        orgstdout = sys.stdout
        orgstderr = sys.stderr
        stdout = sys.stdout = StringIO()
        stderr = sys.stderr = StringIO()

        # this call actually executes the test
        self.orgtest(res)

        sys.stdout = orgstdout
        sys.stderr = orgstderr

        data = None
        if res._status == ResultCollector.SKIP: #needs extra data
          data = {'reason': res._reason}

        self.plugin.sendToMaster(res._status, res._exc_info,
                          (stdout.getvalue(), stderr.getvalue()), data)


class ResultCollector(object):
    'Result collector, shadows the original result object'

    SUCCESS = 0		# Test success
    FAILURE = 1		# Test failure
    ERROR = 2		# Test error
    ABORT = 3		# Error in master/slave protocol, test aborted
    SKIP = 4    # Test was skipped, reason will be given

    def __init__(self, result):
        """
        Initialise the object

        Arguments: result - the original result object
        Returns:   None
        """
        self._result = result
        self._status = None
        self._exc_info = (None, None)
        self._reason = None

    def addSuccess(self, test):
        """
        Add success status

        Arguments: test - the test objective
        Returns:   None
        """
        self._result.addSuccess(test)
        self._status = self.SUCCESS

    def addFailure(self, test, exc_info):
        """
        Add failure status

        Arguments: test - the test objective
        Returns:   None
        """
        self._result.addFailure(test, exc_info)
        self._status = self.FAILURE
        self._exc_info = exc_info[1], Traceback.make(exc_info[2])

    def addError(self, test, exc_info):
        """
        Add error status

        Arguments: test - the test objective
        Returns:   None
        """
        self._result.addError(test, exc_info)
        self._status = self.ERROR
        self._exc_info = exc_info[1], Traceback.make(exc_info[2])

    def addSkip(self, test, reason):
        """
        Add skip status

        Arguments: test - the test objective
                   reason - the reason the test was skipped
        Returns:   None
        """
        self._result.addSkip(test, reason)
        self._status = self.SKIP
        self._reason = reason

    def __getattr__(self, attr):
        """
        Mirror result object attributes.

        Arguments: attr - attribute to retrieve
        Returns:   Attribute data
        """
        return getattr(self._result, attr)

    def __setattr__(self, attr, value):
        """
        Mirror result object attributes.

        Arguments: attr - attribute to set
                   value - value to set
        Returns:   None
        """
        if attr in ('_status', '_result', '_exc_info'):
            super(ResultCollector, self).__setattr__(attr, value)
        else:
            setattr(self._result, attr, value)
