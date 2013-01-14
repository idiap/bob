/**
 * @file visioner/programs/vgui/controls.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef _ACC_GUI_CONTROLS_H_
#define _ACC_GUI_CONTROLS_H_

#include <QtGui>

/**
 * Widgets
 */

// Updates a list of widgets
void updateBuddies(std::vector<QWidget*>& buddies);

// Create a tool button using the icon file path
QToolButton* toolButton(const QString& iconPath, const QString& toolTip, int size = 16);

// Create a push button using the text label and the icon file path
QPushButton* pushButton(const QString& text, const QString& iconPath, const QString& toolTip);

// Create a menu action
QAction* action(const QString& iconPath, const QString& text, bool checkable);

// Create a separator for a dialog
QFrame* hSeparator();
QFrame* vSeparator();

// Build the OK/Cancel layout
QHBoxLayout* buildOKCancelLayout(QDialog* parent);

// Functions for fast horizontal layout building
// NB: if some widget/layout is <0> it will be replaced by <addStretch>

// Build a horizontal layout with two elements
template <	typename LeftWidget,
		typename RightWidget>
QHBoxLayout* buildHLayout(	LeftWidget* leftWidget,
					RightWidget* rightWidget,
					int leftStretch = 40,
					int rightStretch = 60)
{
	QHBoxLayout* hLayout = new QHBoxLayout();
	addToHLayout(hLayout, leftWidget, leftStretch);
	addToHLayout(hLayout, rightWidget, rightStretch);
	hLayout->setMargin(0);
	return hLayout;
}

// Build a horizontal layout with three elements
template <	typename LeftWidget,
		typename MiddleWidget,
		typename RightWidget>
QHBoxLayout* buildHLayout(	LeftWidget* leftWidget,
					MiddleWidget* middleWidget,
					RightWidget* rightWidget,
					int leftStretch = 40,
					int middleStretch = 30,
					int rightStretch = 30)
{
	QHBoxLayout* hLayout = new QHBoxLayout();
	addToHLayout(hLayout, leftWidget, leftStretch);
	addToHLayout(hLayout, middleWidget, middleStretch);
	addToHLayout(hLayout, rightWidget, rightStretch);
	hLayout->setMargin(0);
	return hLayout;
}

// Build a horizontal layout with four elements
template <	typename LeftWidget,
		typename Middle1Widget,
		typename Middle2Widget,
		typename RightWidget>
QHBoxLayout* buildHLayout(	LeftWidget* leftWidget,
					Middle1Widget* middle1Widget,
					Middle2Widget* middle2Widget,
					RightWidget* rightWidget,
					int leftStretch = 40,
					int middle1Stretch = 20,
					int middle2Stretch = 20,
					int rightStretch = 20)
{
	QHBoxLayout* hLayout = new QHBoxLayout();
	addToHLayout(hLayout, leftWidget, leftStretch);
	addToHLayout(hLayout, middle1Widget, middle1Stretch);
	addToHLayout(hLayout, middle2Widget, middle2Stretch);
	addToHLayout(hLayout, rightWidget, rightStretch);
	hLayout->setMargin(0);
	return hLayout;
}

// Add a component to some horizontal layout
// NB: if some widget/layout is <0> it will be replaced by <addStretch>
void addToHLayout(QHBoxLayout* hLayout, QWidget* widget, int stretchFactor);
void addToHLayout(QHBoxLayout* hLayout, QLayout* layout, int stretchFactor);

// Build an icon used for displaying the current chosen color
QPixmap color_pixmap(const QColor& color, int w = 24, int h = 24);

// FontColorButton - Qt button used for displaying and selecting some font and color
class FontColorButton : public QPushButton
{
public:

	// Constructor
	FontColorButton(QFont& font, QColor& color, QWidget* parent = 0)
		: QPushButton("", parent), m_font(font), m_color(color)
	{
	}

	// Add a buddy widget to be updated
	bool addUpdateBuddy(QWidget* buddy)
	{
		if (buddy != 0)
			m_buddies.push_back(buddy);
		return buddy != 0;
	}

protected:

	// Catch paint event
	void paintEvent(QPaintEvent* event);

	// Catch the click event
	void mousePressEvent(QMouseEvent* event);

private:

	// Attributes
	QFont&			m_font;		// The stored font to update
	QColor&			m_color;	// The stored color only to display
	std::vector<QWidget*>	m_buddies;	// Buddies to update when the font is changed
};

// ColorButton - Qt button used for displaying and selecting some color
class ColorButton : public QPushButton
{
public:

	// Constructor
	ColorButton(QColor& color, QWidget* parent = 0)
		: QPushButton("", parent), m_color(color)
	{
	}

	// Add a buddy widget to be updated
	bool addUpdateBuddy(QWidget* buddy)
	{
		if (buddy != 0)
			m_buddies.push_back(buddy);
		return buddy != 0;
	}

protected:

	// Catch the click event
	void mousePressEvent(QMouseEvent* event);

	// Catch paint event
	void paintEvent(QPaintEvent* event);

private:

	// Attributes
	QColor&			m_color;	// The stored color to update
	std::vector<QWidget*>	m_buddies;	// Buddies to update when the color is changed
};

// ValueCombo - Qt combobox used for setting some value
template <typename TValue>
class ValueCombo : public QComboBox
{
public:

	typedef typename std::vector<TValue>::const_iterator TValueConstIt;

	// Constructor
	ValueCombo(TValue& value, QWidget* parent = 0)
		: QComboBox(parent), m_value(value), m_lastIndex(-1)
	{
	}

	// Add an item (text + value)
	void add(const QString& text, TValue value)
	{
		addItem(text);
		m_options.push_back(value);

		// Make sure the selection is set
		for (TValueConstIt it = m_options.begin(); it != m_options.end(); ++ it)
			if (*it == m_value)
		{
			m_lastIndex = it - m_options.begin();
			setCurrentIndex(m_lastIndex);
			break;
		}
	}

	// Add a buddy widget to be updated
	bool addUpdateBuddy(QWidget* buddy)
	{
		if (buddy != 0)
		{
			m_buddies.push_back(buddy);
		}
		return buddy != 0;
	}

protected:

	// Catch the selection change event
	void paintEvent(QPaintEvent* event)
	{
		QComboBox::paintEvent(event);

		int index = currentIndex();
		if (index >= 0 && index < (int)m_options.size() && index != m_lastIndex)
		{
			m_value = m_options[index];
			m_lastIndex = index;
			updateBuddies(m_buddies);
		}
	}

private:

	// Attributes
	TValue&			m_value;	// The stored value to update
	int			m_lastIndex;	// Keep track of the last selected index
	std::vector<TValue>	m_options;	// Options values that can be used
	std::vector<QWidget*>	m_buddies;	// Buddies to update when the value is changed
};

// ValueCheck - Qt checkbox used for setting some value
class ValueCheck : public QCheckBox
{
public:

	// Constructor
	ValueCheck(bool& value, const QString& text, QWidget* parent = 0)
		: QCheckBox(text, parent),  m_painted(false), m_value(value)
	{
		setCheckable(true);
		setTristate(false);
	}


protected:

	// Catch the checked change event
	void paintEvent(QPaintEvent* event);

private:

	// Attributes
	bool			m_painted;
	bool&			m_value;		// The stored value to update
};

// ExtendedCheckBox - the same functionality as ValueCheck, but there is no need to modify some stored parameter
class ExtendedCheckBox : public ValueCheck
{
public:

	// Constructor
	ExtendedCheckBox(bool value, const QString& text, QWidget* parent = 0)
		: ValueCheck(m_dummy, text, parent), m_dummy(value)
	{
		setChecked(m_dummy);
	}

private:

	// Attributes
	bool			m_dummy;		// Dummy value to be modified by ValueCheck
};

// ExtendedCheckBox - the same functionality as ValueCheck, but there is no need to modify some stored parameter
class ExtendedRadioButton : public QRadioButton
{
public:

	// Constructor
	ExtendedRadioButton(const QString& text, QWidget* parent = 0)
		: QRadioButton(text, parent), m_painted(false), m_checked(false)
	{
	}

protected:

	// Catch the checked change event
	void paintEvent(QPaintEvent* event);

private:

	// Attributes
	bool				m_painted;
	bool				m_checked;
};

// RenameTabBar - custom tab bar with renaming support (by double clicking it)
class RenameTabBar : public QTabBar
{
public:

	// Constructor
	RenameTabBar(QWidget* parent = 0);

protected:

	// Catch the double click event to handle renaming
	void mouseDoubleClickEvent(QMouseEvent* event);
};

class RenameTabWidget : public QTabWidget
{
public:

	// Constructor
	RenameTabWidget(QWidget* parent = 0);
};

// Custom gradient label
class GradientLabel : public QLabel
{
public:

	// Constructor
	GradientLabel(const QColor& colorMin, const QColor& colorMax, QWidget* parent = 0);

protected:

	// Catch the paint event - draw the gradient
	void paintEvent(QPaintEvent* event);

private:

	// Attributes
	QColor		m_colorMin, m_colorMax;
};

#endif
