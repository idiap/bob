#
# Add the source files to the list of files to be added to the library
#
PROJECT(core)
set(THIS core)

SET(${THIS}_SRC
		"general.cc"
		"VariableCollector.cc"
		"Object.cc"
		"Parameters.cc"
		"File.cc"
		"spCore.cc"
		"spCoreChain.cc"
		"ipCore.cc"
		"apCore.cc"
		"DataSet.cc"
		"MemoryDataSet.cc"
		"DiskDataSet.cc"
		"ImageScanDataSet.cc"
		"Trainer.cc"
		"Machine.cc"
		"Tensor.cc"
		"TensorFile.cc"
		"TensorList.cc"
		"TensorListTarget.cc"
		"MTimer.cc"
		"CmdOption.cc"
		"FileListCmdOption.cc"
		"FileList.cc"
		"CmdLine.cc"
		"CmdFile.cc"
	CACHE INTERNAL "sources")

include(../SetupPackage.cmake)
