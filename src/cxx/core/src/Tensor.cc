/**
 * @file cxx/core/src/Tensor.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the Tensor(Wrapper) class
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

//#include "core/array_common.h"
#include "core/Tensor.h"

namespace bob {

const char *str_TensorTypeName[] = {"char", "short", "int", "long", "float", "double"};

}

void bob::Tensor::raiseError(std::string msg) const {
  std::cerr << "Error: " << msg << std::endl;
}

void bob::Tensor::raiseFatalError(std::string msg) const {
  std::cerr << "Fatal Error: " << msg << std::endl;
  exit(-1);
}

void bob::Tensor::setTensor( const bob::Tensor *src) {
  bob::Tensor::Type type = m_datatype;
  bob::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::string msg("bob::Tensor::setTensor() don't know how to set a Tensor from a different type. Try a copy instead.");
    raiseError(msg);
    return;
  }

  switch(src_type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* src_char;
      bob::CharTensor* this_char;
      src_char = dynamic_cast<const bob::CharTensor*>(src);
      this_char = dynamic_cast<bob::CharTensor*>(this);
      this_char->setTensor( src_char);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* src_short;
      bob::ShortTensor* this_short;
      src_short = dynamic_cast<const bob::ShortTensor*>(src);
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      this_short->setTensor( src_short);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* src_int;
      bob::IntTensor* this_int;
      src_int = dynamic_cast<const bob::IntTensor*>(src);
      this_int = dynamic_cast<bob::IntTensor*>(this);
      this_int->setTensor( src_int);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* src_long;
      bob::LongTensor* this_long;
      src_long = dynamic_cast<const bob::LongTensor*>(src);
      this_long = dynamic_cast<bob::LongTensor*>(this);
      this_long->setTensor( src_long);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* src_float;
      bob::FloatTensor* this_float;
      src_float = dynamic_cast<const bob::FloatTensor*>(src);
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      this_float->setTensor( src_float);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* src_double;
      bob::DoubleTensor* this_double;
      src_double = dynamic_cast<const bob::DoubleTensor*>(src);
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      this_double->setTensor( src_double);
      break;
    case bob::Tensor::Undefined:
    default:
      std::string msg("Tensor::setTensor() don't know how to set a Tensor from a Undefined/Unknown type.");
      raiseError(msg);
      return;
  }
}

void bob::Tensor::copy(const bob::Tensor *src) {
  bob::Tensor::Type type = m_datatype;
  bob::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type == bob::Tensor::Undefined || type == bob::Tensor::Undefined )
  {
    std::cerr << "Error: Tensor::copy() don't know how to copy from or to an \"Undefined type\" Tensor." << std::endl;
    return;
  }

  switch(src_type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* src_char;
      src_char = dynamic_cast<const bob::CharTensor*>(src);        
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_char);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_char);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_char);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_char);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_char);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_char);
          break;
        default:
          return;
      }
      break;

    case bob::Tensor::Short:
      const bob::ShortTensor* src_short;
      src_short = dynamic_cast<const bob::ShortTensor*>(src);
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_short);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_short);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_short);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_short);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_short);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_short);
          break;
        default:
          return;
      }
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* src_int;
      src_int = dynamic_cast<const bob::IntTensor*>(src);
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_int);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_int);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_int);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_int);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_int);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_int);
          break;
        default:
          return;
      }
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* src_long;
      src_long = dynamic_cast<const bob::LongTensor*>(src);
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_long);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_long);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_long);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_long);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_long);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_long);
          break;
        default:
          return;
      }
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* src_float;
      src_float = dynamic_cast<const bob::FloatTensor*>(src);
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_float);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_float);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_float);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_float);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_float);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_float);
          break;
        default:
          return;
      }
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* src_double;
      src_double = dynamic_cast<const bob::DoubleTensor*>(src);
      switch(type)
      {
        case bob::Tensor::Char:
          bob::CharTensor* this_char;
          this_char = dynamic_cast<bob::CharTensor*>(this);
          this_char->copy( src_double);
          break;
        case bob::Tensor::Short:
          bob::ShortTensor* this_short;
          this_short = dynamic_cast<bob::ShortTensor*>(this);
          this_short->copy( src_double);
          break;
        case bob::Tensor::Int:
          bob::IntTensor* this_int;
          this_int = dynamic_cast<bob::IntTensor*>(this);
          this_int->copy( src_double);
          break;
        case bob::Tensor::Long:
          bob::LongTensor* this_long;
          this_long = dynamic_cast<bob::LongTensor*>(this);
          this_long->copy( src_double);
          break;
        case bob::Tensor::Float:
          bob::FloatTensor* this_float;
          this_float = dynamic_cast<bob::FloatTensor*>(this);
          this_float->copy( src_double);
          break;
        case bob::Tensor::Double:
          bob::DoubleTensor* this_double;
          this_double = dynamic_cast<bob::DoubleTensor*>(this);
          this_double->copy( src_double);
          break;
        default:
          return;
      }
      break;
    case bob::Tensor::Undefined:
      std::cerr << "Error: bob::Tensor::copy() don't know how to set a bob::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void bob::Tensor::transpose( const bob::Tensor *src, int dimension1, int dimension2) {
  bob::Tensor::Type type = m_datatype;
  bob::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: bob::Tensor::transpose() don't know how to set a bob::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* src_char;
      bob::CharTensor* this_char;
      src_char = dynamic_cast<const bob::CharTensor*>(src);
      this_char = dynamic_cast<bob::CharTensor*>(this);
      this_char->transpose( src_char, dimension1, dimension2);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* src_short;
      bob::ShortTensor* this_short;
      src_short = dynamic_cast<const bob::ShortTensor*>(src);
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      this_short->transpose( src_short, dimension1, dimension2);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* src_int;
      bob::IntTensor* this_int;
      src_int = dynamic_cast<const bob::IntTensor*>(src);
      this_int = dynamic_cast<bob::IntTensor*>(this);
      this_int->transpose( src_int, dimension1, dimension2);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* src_long;
      bob::LongTensor* this_long;
      src_long = dynamic_cast<const bob::LongTensor*>(src);
      this_long = dynamic_cast<bob::LongTensor*>(this);
      this_long->transpose( src_long, dimension1, dimension2);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* src_float;
      bob::FloatTensor* this_float;
      src_float = dynamic_cast<const bob::FloatTensor*>(src);
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      this_float->transpose( src_float, dimension1, dimension2);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* src_double;
      bob::DoubleTensor* this_double;
      src_double = dynamic_cast<const bob::DoubleTensor*>(src);
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      this_double->transpose( src_double, dimension1, dimension2);
      break;
    case bob::Tensor::Undefined:
      std::cerr << "Error: bob::Tensor::transpose() don't know how to set a bob::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void bob::Tensor::narrow (const bob::Tensor *src, int dimension, long firstIndex,
    long size) {
  bob::Tensor::Type type = m_datatype;
  bob::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: bob::Tensor::narrow() don't know how to set a bob::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* src_char;
      bob::CharTensor* this_char;
      src_char = dynamic_cast<const bob::CharTensor*>(src);
      this_char = dynamic_cast<bob::CharTensor*>(this);
      this_char->narrow( src_char, dimension, firstIndex, size);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* src_short;
      bob::ShortTensor* this_short;
      src_short = dynamic_cast<const bob::ShortTensor*>(src);
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      this_short->narrow( src_short, dimension, firstIndex, size);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* src_int;
      bob::IntTensor* this_int;
      src_int = dynamic_cast<const bob::IntTensor*>(src);
      this_int = dynamic_cast<bob::IntTensor*>(this);
      this_int->narrow( src_int, dimension, firstIndex, size);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* src_long;
      bob::LongTensor* this_long;
      src_long = dynamic_cast<const bob::LongTensor*>(src);
      this_long = dynamic_cast<bob::LongTensor*>(this);
      this_long->narrow( src_long, dimension, firstIndex, size);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* src_float;
      bob::FloatTensor* this_float;
      src_float = dynamic_cast<const bob::FloatTensor*>(src);
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      this_float->narrow( src_float, dimension, firstIndex, size);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* src_double;
      bob::DoubleTensor* this_double;
      src_double = dynamic_cast<const bob::DoubleTensor*>(src);
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      this_double->narrow( src_double, dimension, firstIndex, size);
      break;
    case bob::Tensor::Undefined:
      std::cerr << "Error: bob::Tensor::narrow() don't know how to set a bob::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void bob::Tensor::select( const bob::Tensor* src, int dimension, long sliceIndex) {
  bob::Tensor::Type type = m_datatype;
  bob::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: bob::Tensor::select() don't know how to set a bob::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* src_char;
      bob::CharTensor* this_char;
      src_char = dynamic_cast<const bob::CharTensor*>(src);
      this_char = dynamic_cast<bob::CharTensor*>(this);
      this_char->select( src_char, dimension, sliceIndex);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* src_short;
      bob::ShortTensor* this_short;
      src_short = dynamic_cast<const bob::ShortTensor*>(src);
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      this_short->select( src_short, dimension, sliceIndex);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* src_int;
      bob::IntTensor* this_int;
      src_int = dynamic_cast<const bob::IntTensor*>(src);
      this_int = dynamic_cast<bob::IntTensor*>(this);
      this_int->select( src_int, dimension, sliceIndex);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* src_long;
      bob::LongTensor* this_long;
      src_long = dynamic_cast<const bob::LongTensor*>(src);
      this_long = dynamic_cast<bob::LongTensor*>(this);
      this_long->select( src_long, dimension, sliceIndex);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* src_float;
      bob::FloatTensor* this_float;
      src_float = dynamic_cast<const bob::FloatTensor*>(src);
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      this_float->select( src_float, dimension, sliceIndex);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* src_double;
      bob::DoubleTensor* this_double;
      src_double = dynamic_cast<const bob::DoubleTensor*>(src);
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      this_double->select( src_double, dimension, sliceIndex);
      break;
    case bob::Tensor::Undefined:
      std::cerr << "Error: bob::Tensor::select() don't know how to set a bob::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

bob::Tensor* bob::Tensor::select( int dimension, long sliceIndex) const {
  bob::Tensor* res = 0;
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* this_char;
      this_char = dynamic_cast<const bob::CharTensor*>(this);
      res = this_char->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* this_short;
      this_short = dynamic_cast<const bob::ShortTensor*>(this);
      res = this_short->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* this_int;
      this_int = dynamic_cast<const bob::IntTensor*>(this);
      res = this_int->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* this_long;
      this_long = dynamic_cast<const bob::LongTensor*>(this);
      res = this_long->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* this_float;
      this_float = dynamic_cast<const bob::FloatTensor*>(this);
      res = this_float->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* this_double;
      this_double = dynamic_cast<const bob::DoubleTensor*>(this);
      res = this_double->select( dimension, sliceIndex);
      break;
    case bob::Tensor::Undefined:
      std::cerr << "Error: bob::Tensor::select() don't know how to set a bob::Tensor from Undefined type." << std::endl;
    default:
      break;
  }
  return res;
}

int bob::Tensor::typeSize() const {
  int res = 0;
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      res = sizeof(char);
      break;
    case bob::Tensor::Short:
      res = sizeof(short);
      break;
    case bob::Tensor::Int:
      res = sizeof(int);
      break;
    case bob::Tensor::Long:
      res = sizeof(long);
      break;
    case bob::Tensor::Float:
      res = sizeof(float);
      break;
    case bob::Tensor::Double:
      res = sizeof(double);
      break;
    case bob::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

const void* bob::Tensor::dataR() const {
  const void* res = 0;
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* this_char;
      this_char = dynamic_cast<const bob::CharTensor*>(this);
      res = this_char->dataR();
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* this_short;
      this_short = dynamic_cast<const bob::ShortTensor*>(this);
      res = this_short->dataR();
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* this_int;
      this_int = dynamic_cast<const bob::IntTensor*>(this);
      res = this_int->dataR();
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* this_long;
      this_long = dynamic_cast<const bob::LongTensor*>(this);
      res = this_long->dataR();
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* this_float;
      this_float = dynamic_cast<const bob::FloatTensor*>(this);
      res = this_float->dataR();
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* this_double;
      this_double = dynamic_cast<const bob::DoubleTensor*>(this);
      res = this_double->dataR();
      break;
    case bob::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

void* bob::Tensor::dataW() {
  void* res = 0;
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      bob::CharTensor* this_char;
      this_char = dynamic_cast<bob::CharTensor*>(this);
      res = this_char->dataW();
      break;
    case bob::Tensor::Short:
      bob::ShortTensor* this_short;
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      res = this_short->dataW();
      break;
    case bob::Tensor::Int:
      bob::IntTensor* this_int;
      this_int = dynamic_cast<bob::IntTensor*>(this);
      res = this_int->dataW();
      break;
    case bob::Tensor::Long:
      bob::LongTensor* this_long;
      this_long = dynamic_cast<bob::LongTensor*>(this);
      res = this_long->dataW();
      break;
    case bob::Tensor::Float:
      bob::FloatTensor* this_float;
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      res = this_float->dataW();
      break;
    case bob::Tensor::Double:
      bob::DoubleTensor* this_double;
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      res = this_double->dataW();
      break;
    case bob::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

void bob::Tensor::resetFromData() {
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      bob::CharTensor* this_char;
      this_char = dynamic_cast<bob::CharTensor*>(this);
      this_char->resetFromData();
      break;
    case bob::Tensor::Short:
      bob::ShortTensor* this_short;
      this_short = dynamic_cast<bob::ShortTensor*>(this);
      this_short->resetFromData();
      break;
    case bob::Tensor::Int:
      bob::IntTensor* this_int;
      this_int = dynamic_cast<bob::IntTensor*>(this);
      this_int->resetFromData();
      break;
    case bob::Tensor::Long:
      bob::LongTensor* this_long;
      this_long = dynamic_cast<bob::LongTensor*>(this);
      this_long->resetFromData();
      break;
    case bob::Tensor::Float:
      bob::FloatTensor* this_float;
      this_float = dynamic_cast<bob::FloatTensor*>(this);
      this_float->resetFromData();
      break;
    case bob::Tensor::Double:
      bob::DoubleTensor* this_double;
      this_double = dynamic_cast<bob::DoubleTensor*>(this);
      this_double->resetFromData();
      break;
    case bob::Tensor::Undefined:
    default:
      break;
  }
}

long bob::Tensor::stride(int dim) const {
  long res = 0;
  bob::Tensor::Type type = m_datatype;

  switch(type)
  {
    case bob::Tensor::Char:
      const bob::CharTensor* this_char;
      this_char = dynamic_cast<const bob::CharTensor*>(this);
      res = this_char->stride(dim);
      break;
    case bob::Tensor::Short:
      const bob::ShortTensor* this_short;
      this_short = dynamic_cast<const bob::ShortTensor*>(this);
      res = this_short->stride(dim);
      break;
    case bob::Tensor::Int:
      const bob::IntTensor* this_int;
      this_int = dynamic_cast<const bob::IntTensor*>(this);
      res = this_int->stride(dim);
      break;
    case bob::Tensor::Long:
      const bob::LongTensor* this_long;
      this_long = dynamic_cast<const bob::LongTensor*>(this);
      res = this_long->stride(dim);
      break;
    case bob::Tensor::Float:
      const bob::FloatTensor* this_float;
      this_float = dynamic_cast<const bob::FloatTensor*>(this);
      res = this_float->stride(dim);
      break;
    case bob::Tensor::Double:
      const bob::DoubleTensor* this_double;
      this_double = dynamic_cast<const bob::DoubleTensor*>(this);
      res = this_double->stride(dim);
      break;
    case bob::Tensor::Undefined:
    default:
      break;
  }
  return res;
}
