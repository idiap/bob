project(core)
cmake_minimum_required(VERSION 2.4)

include(../macros.cmake)

SET(src
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

set(this core)
set(deps torch_th)
torch_shlib(torch_${this} ${src} ${deps} "" ${INSTALL_DIR}/lib)
torch_archive(torch_${this} ${src} ${deps} "" ${INSTALL_DIR}/lib)

file(COPY ./ DESTINATION ${INCLUDE_DIR}/torch/${this} FILES_MATCHING PATTERN "*.h")
