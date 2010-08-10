project(core)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps th) #other torch subprojects
set(shared "") #shared externals to link against (link)

# This defines the list of source files inside this package.
set(src
		"general.cc"
		"VariableCollector.cc"
		"Object.cc"
		"Parameters.cc"
		"File.cc"
		"spCore.cc"
		"spCoreChain.cc"
#		"apCore.cc"
		"DataSet.cc"
		"MemoryDataSet.cc"
		"DiskDataSet.cc"
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
		"CmdFile.cc")

include(../macros.cmake)
torch_package(core "${src}" "${deps}" "${shared}")
