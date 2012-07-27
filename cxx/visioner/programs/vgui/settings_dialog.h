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
