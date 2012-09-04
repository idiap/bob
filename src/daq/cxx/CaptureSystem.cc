/**
 * @file cxx/daq/src/CaptureSystem.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "bob/daq/CaptureSystem.h"

#include <cstdio>

//#include "daq/OpenCVDisplay.h"
//#include "daq/OpenCVFaceLocalization.h"
#include "bob/daq/VisionerFaceLocalization.h"
#include "bob/daq/Controller.h"
#include "bob/daq/SimpleController.h"
#include "bob/daq/V4LCamera.h"
//#include "daq/OpenCVOutputWriter.h"
#include "bob/daq/QtDisplay.h"
#include "bob/daq/BobOutputWriter.h"

namespace bob { namespace daq {

CaptureSystem::CaptureSystem(boost::shared_ptr<Camera> camera, const char* faceLocalizationModelPath) {
  this->recordingDelay = 3;
  this->length = 10;
  this->camera = camera;
  this->outputDir = ".";
  this->outputName = "output";
  this->thumbnail = "";
  this->fullscreen = false;
  this->displayWidth = -1;
  this->displayHeight = -1;
  this->faceLocalizationModelPath = faceLocalizationModelPath;
}

CaptureSystem::~CaptureSystem() {
}

void CaptureSystem::start() {
  int i = 0;
  QApplication app(i, NULL);
  
  //OpenCVDisplay display;
  QtDisplay display;

  //OpenCVFaceLocalization fl;
  VisionerFaceLocalization fl(faceLocalizationModelPath);
  SimpleController controller;
  //OpenCVOutputWriter outputWriter;
  BobOutputWriter outputWriter;

  
  controller.addControllerCallback(fl);
  controller.addControllerCallback(display);
  controller.addStoppable(*camera);
  fl.addFaceLocalizationCallback(display);
  display.addKeyPressCallback(controller);
  
  int err = camera->open();
  if (err != 0) {
    fprintf(stderr, "Can't open camera.\n");
    return;
  }
  
  camera->addCameraCallback(controller);

  Camera::FrameSize frameSize = camera->getFrameSize();
  Camera::FrameInterval frameInterval = camera->getFrameInterval();
  outputWriter.setOutputName(outputName);
  outputWriter.setOutputDir(outputDir);
  outputWriter.open(frameSize.width, frameSize.height, frameInterval.denominator/frameInterval.numerator);
  
  controller.setOutputWriter(outputWriter);
  controller.setRecordingDelay(recordingDelay);
  controller.setLength(length);

  display.setThumbnail(thumbnail);
  display.setFullscreen(fullscreen);
  display.setDisplaySize(displayWidth, displayHeight);
  display.setExecuteOnStartRecording(onStartRecording);
  display.setExecuteOnStopRecording(onStopRecording);
  display.setText(text);
  
  err = camera->start();
  
  if (err != 0) {
    fprintf(stderr, "Error starting capture\n");
    return;
  }

  bool ok = fl.start();
  if (!ok) {
    fprintf(stderr, "Error starting face localization\n");
  }

  display.start();
  
  camera->removeCameraCallback(controller);
}

void CaptureSystem::setRecordingDelay(int recordingDelay) {
  this->recordingDelay = recordingDelay;
}

int CaptureSystem::getRecordingDelay() {
  return this->recordingDelay;
}

void CaptureSystem::setLength(int length) {
  this->length = length;
}

int CaptureSystem::getLength() {
  return this->length;
}

void CaptureSystem::setOutputDir(const std::string& dir) {
  this->outputDir = dir;
}

std::string CaptureSystem::getOutputDir() {
  return this->outputDir;
}

void CaptureSystem::setOutputName(const std::string& name) {
  this->outputName = name;
}

std::string CaptureSystem::getOutputName() {
  return this->outputName;
}

void CaptureSystem::setThumbnail(const std::string& path) {
  this->thumbnail = path;
}

std::string CaptureSystem::getThumbnail() {
  return this->thumbnail;
}

void CaptureSystem::setFullScreen(bool fullscreen) {
  this->fullscreen = fullscreen;
}

bool CaptureSystem::getFullScreen() {
  return this->fullscreen;
}

void CaptureSystem::setDisplaySize(int width, int height) {
  this->displayWidth = width;
  this->displayHeight = height;
}

void CaptureSystem::setExecuteOnStartRecording(const std::string& program) {
  onStartRecording = program;
}

void CaptureSystem::setExecuteOnStopRecording(const std::string& program) {
  onStopRecording = program;
}

void CaptureSystem::setText(const std::string& text) {
  this->text = text;
}

}}