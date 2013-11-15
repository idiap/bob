/**
 * @file visioner/programs/vgui/settings_dialog.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include <QtGui>

#include "settings.h"

class ItemSettingsDialog : public QDialog {
  public:

    // Constructor
    ItemSettingsDialog(const ItemSettings& settings, QWidget* parent = 0)
      : QDialog(parent), m_settings(settings)
    {
      buildGUI();
    }

    // Get the modified settings
    const ItemSettings& settings() const { return m_settings; }

  private:

    // GUI construction
    void buildGUI();

  private:

    // Attributes
    ItemSettings		m_settings;		// Settings to modify
};

class SceneSettingsDialog : public QDialog
{
  public:

    // Constructor
    SceneSettingsDialog(const SceneSettings& settings, QWidget* parent = 0)
      : QDialog(parent), m_settings(settings)
    {
      buildGUI();
    }

    // Get the modified settings
    const SceneSettings& settings() const
    { 
      return m_settings;
    }

  private:

    // GUI construction
    void buildGUI();

  private:

    // Attributes
    SceneSettings		m_settings;		// Settings to modify
};

#endif	// SETTINGS_DIALOG_H
