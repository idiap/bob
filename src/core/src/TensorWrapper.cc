/**
 * @file TensorWrapper.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El-Shafey</a> 
 *
 * @brief Implements the TensorWrapper class
 */

#include "core/Tensor.h"

void Torch::Tensor::raiseError(std::string msg) const {
  std::cerr << "Error: " << msg << std::endl;
}

void Torch::Tensor::raiseFatalError(std::string msg) const {
  std::cerr << "Fatal Error: " << msg << std::endl;
  exit(-1);
}

void Torch::Tensor::setTensor( const Torch::Tensor *src) {
  Torch::Tensor::Type type = m_datatype;
  Torch::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::string msg("Torch::Tensor::setTensor() don't know how to set a Tensor from a different type. Try a copy instead.");
    raiseError(msg);
    return;
  }

  switch(src_type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* src_char;
      Torch::CharTensor* this_char;
      src_char = dynamic_cast<const Torch::CharTensor*>(src);
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      this_char->setTensor( src_char);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* src_short;
      Torch::ShortTensor* this_short;
      src_short = dynamic_cast<const Torch::ShortTensor*>(src);
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      this_short->setTensor( src_short);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* src_int;
      Torch::IntTensor* this_int;
      src_int = dynamic_cast<const Torch::IntTensor*>(src);
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      this_int->setTensor( src_int);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* src_long;
      Torch::LongTensor* this_long;
      src_long = dynamic_cast<const Torch::LongTensor*>(src);
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      this_long->setTensor( src_long);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* src_float;
      Torch::FloatTensor* this_float;
      src_float = dynamic_cast<const Torch::FloatTensor*>(src);
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      this_float->setTensor( src_float);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* src_double;
      Torch::DoubleTensor* this_double;
      src_double = dynamic_cast<const Torch::DoubleTensor*>(src);
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      this_double->setTensor( src_double);
      break;
    case Torch::Tensor::Undefined:
    default:
      std::string msg("Tensor::setTensor() don't know how to set a Tensor from a Undefined/Unknown type.");
      raiseError(msg);
      return;
  }
}

void Torch::Tensor::copy(const Torch::Tensor *src) {
  Torch::Tensor::Type type = m_datatype;
  Torch::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type == Torch::Tensor::Undefined || type == Torch::Tensor::Undefined )
  {
    std::cerr << "Error: Tensor::copy() don't know how to copy from or to an \"Undefined type\" Tensor." << std::endl;
    return;
  }

  switch(src_type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* src_char;
      src_char = dynamic_cast<const Torch::CharTensor*>(src);        
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_char);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_char);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_char);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_char);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_char);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_char);
          break;
        default:
          return;
      }
      break;

    case Torch::Tensor::Short:
      const Torch::ShortTensor* src_short;
      src_short = dynamic_cast<const Torch::ShortTensor*>(src);
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_short);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_short);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_short);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_short);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_short);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_short);
          break;
        default:
          return;
      }
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* src_int;
      src_int = dynamic_cast<const Torch::IntTensor*>(src);
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_int);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_int);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_int);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_int);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_int);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_int);
          break;
        default:
          return;
      }
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* src_long;
      src_long = dynamic_cast<const Torch::LongTensor*>(src);
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_long);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_long);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_long);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_long);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_long);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_long);
          break;
        default:
          return;
      }
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* src_float;
      src_float = dynamic_cast<const Torch::FloatTensor*>(src);
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_float);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_float);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_float);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_float);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_float);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_float);
          break;
        default:
          return;
      }
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* src_double;
      src_double = dynamic_cast<const Torch::DoubleTensor*>(src);
      switch(type)
      {
        case Torch::Tensor::Char:
          Torch::CharTensor* this_char;
          this_char = dynamic_cast<Torch::CharTensor*>(this);
          this_char->copy( src_double);
          break;
        case Torch::Tensor::Short:
          Torch::ShortTensor* this_short;
          this_short = dynamic_cast<Torch::ShortTensor*>(this);
          this_short->copy( src_double);
          break;
        case Torch::Tensor::Int:
          Torch::IntTensor* this_int;
          this_int = dynamic_cast<Torch::IntTensor*>(this);
          this_int->copy( src_double);
          break;
        case Torch::Tensor::Long:
          Torch::LongTensor* this_long;
          this_long = dynamic_cast<Torch::LongTensor*>(this);
          this_long->copy( src_double);
          break;
        case Torch::Tensor::Float:
          Torch::FloatTensor* this_float;
          this_float = dynamic_cast<Torch::FloatTensor*>(this);
          this_float->copy( src_double);
          break;
        case Torch::Tensor::Double:
          Torch::DoubleTensor* this_double;
          this_double = dynamic_cast<Torch::DoubleTensor*>(this);
          this_double->copy( src_double);
          break;
        default:
          return;
      }
      break;
    case Torch::Tensor::Undefined:
      std::cerr << "Error: Torch::Tensor::copy() don't know how to set a Torch::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void Torch::Tensor::transpose( const Torch::Tensor *src, int dimension1, int dimension2) {
  Torch::Tensor::Type type = m_datatype;
  Torch::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: Torch::Tensor::transpose() don't know how to set a Torch::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* src_char;
      Torch::CharTensor* this_char;
      src_char = dynamic_cast<const Torch::CharTensor*>(src);
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      this_char->transpose( src_char, dimension1, dimension2);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* src_short;
      Torch::ShortTensor* this_short;
      src_short = dynamic_cast<const Torch::ShortTensor*>(src);
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      this_short->transpose( src_short, dimension1, dimension2);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* src_int;
      Torch::IntTensor* this_int;
      src_int = dynamic_cast<const Torch::IntTensor*>(src);
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      this_int->transpose( src_int, dimension1, dimension2);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* src_long;
      Torch::LongTensor* this_long;
      src_long = dynamic_cast<const Torch::LongTensor*>(src);
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      this_long->transpose( src_long, dimension1, dimension2);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* src_float;
      Torch::FloatTensor* this_float;
      src_float = dynamic_cast<const Torch::FloatTensor*>(src);
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      this_float->transpose( src_float, dimension1, dimension2);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* src_double;
      Torch::DoubleTensor* this_double;
      src_double = dynamic_cast<const Torch::DoubleTensor*>(src);
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      this_double->transpose( src_double, dimension1, dimension2);
      break;
    case Torch::Tensor::Undefined:
      std::cerr << "Error: Torch::Tensor::transpose() don't know how to set a Torch::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void Torch::Tensor::narrow (const Torch::Tensor *src, int dimension, long firstIndex,
    long size) {
  Torch::Tensor::Type type = m_datatype;
  Torch::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: Torch::Tensor::narrow() don't know how to set a Torch::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* src_char;
      Torch::CharTensor* this_char;
      src_char = dynamic_cast<const Torch::CharTensor*>(src);
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      this_char->narrow( src_char, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* src_short;
      Torch::ShortTensor* this_short;
      src_short = dynamic_cast<const Torch::ShortTensor*>(src);
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      this_short->narrow( src_short, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* src_int;
      Torch::IntTensor* this_int;
      src_int = dynamic_cast<const Torch::IntTensor*>(src);
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      this_int->narrow( src_int, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* src_long;
      Torch::LongTensor* this_long;
      src_long = dynamic_cast<const Torch::LongTensor*>(src);
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      this_long->narrow( src_long, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* src_float;
      Torch::FloatTensor* this_float;
      src_float = dynamic_cast<const Torch::FloatTensor*>(src);
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      this_float->narrow( src_float, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* src_double;
      Torch::DoubleTensor* this_double;
      src_double = dynamic_cast<const Torch::DoubleTensor*>(src);
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      this_double->narrow( src_double, dimension, firstIndex, size);
      break;
    case Torch::Tensor::Undefined:
      std::cerr << "Error: Torch::Tensor::narrow() don't know how to set a Torch::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

void Torch::Tensor::select( const Torch::Tensor* src, int dimension, long sliceIndex) {
  Torch::Tensor::Type type = m_datatype;
  Torch::Tensor::Type src_type = src->getDatatype();
  // This seems to be reasonable (force the type to be specified before calling the function)
  if( src_type != type )
  {
    std::cerr << "Error: Torch::Tensor::select() don't know how to set a Torch::Tensor from a different type. Try a copy instead." << std::endl;
    return;
  }

  switch(src_type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* src_char;
      Torch::CharTensor* this_char;
      src_char = dynamic_cast<const Torch::CharTensor*>(src);
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      this_char->select( src_char, dimension, sliceIndex);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* src_short;
      Torch::ShortTensor* this_short;
      src_short = dynamic_cast<const Torch::ShortTensor*>(src);
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      this_short->select( src_short, dimension, sliceIndex);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* src_int;
      Torch::IntTensor* this_int;
      src_int = dynamic_cast<const Torch::IntTensor*>(src);
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      this_int->select( src_int, dimension, sliceIndex);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* src_long;
      Torch::LongTensor* this_long;
      src_long = dynamic_cast<const Torch::LongTensor*>(src);
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      this_long->select( src_long, dimension, sliceIndex);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* src_float;
      Torch::FloatTensor* this_float;
      src_float = dynamic_cast<const Torch::FloatTensor*>(src);
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      this_float->select( src_float, dimension, sliceIndex);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* src_double;
      Torch::DoubleTensor* this_double;
      src_double = dynamic_cast<const Torch::DoubleTensor*>(src);
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      this_double->select( src_double, dimension, sliceIndex);
      break;
    case Torch::Tensor::Undefined:
      std::cerr << "Error: Torch::Tensor::select() don't know how to set a Torch::Tensor from Undefined type." << std::endl;
    default:
      return;
  }
}

Torch::Tensor* Torch::Tensor::select( int dimension, long sliceIndex) const {
  Torch::Tensor* res = 0;
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* this_char;
      this_char = dynamic_cast<const Torch::CharTensor*>(this);
      res = this_char->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* this_short;
      this_short = dynamic_cast<const Torch::ShortTensor*>(this);
      res = this_short->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* this_int;
      this_int = dynamic_cast<const Torch::IntTensor*>(this);
      res = this_int->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* this_long;
      this_long = dynamic_cast<const Torch::LongTensor*>(this);
      res = this_long->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* this_float;
      this_float = dynamic_cast<const Torch::FloatTensor*>(this);
      res = this_float->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* this_double;
      this_double = dynamic_cast<const Torch::DoubleTensor*>(this);
      res = this_double->select( dimension, sliceIndex);
      break;
    case Torch::Tensor::Undefined:
      std::cerr << "Error: Torch::Tensor::select() don't know how to set a Torch::Tensor from Undefined type." << std::endl;
    default:
      break;
  }
  return res;
}

int Torch::Tensor::typeSize() const {
  int res = 0;
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      res = sizeof(char);
      break;
    case Torch::Tensor::Short:
      res = sizeof(short);
      break;
    case Torch::Tensor::Int:
      res = sizeof(int);
      break;
    case Torch::Tensor::Long:
      res = sizeof(long);
      break;
    case Torch::Tensor::Float:
      res = sizeof(float);
      break;
    case Torch::Tensor::Double:
      res = sizeof(double);
      break;
    case Torch::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

const void* Torch::Tensor::dataR() const {
  const void* res = 0;
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* this_char;
      this_char = dynamic_cast<const Torch::CharTensor*>(this);
      res = this_char->dataR();
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* this_short;
      this_short = dynamic_cast<const Torch::ShortTensor*>(this);
      res = this_short->dataR();
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* this_int;
      this_int = dynamic_cast<const Torch::IntTensor*>(this);
      res = this_int->dataR();
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* this_long;
      this_long = dynamic_cast<const Torch::LongTensor*>(this);
      res = this_long->dataR();
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* this_float;
      this_float = dynamic_cast<const Torch::FloatTensor*>(this);
      res = this_float->dataR();
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* this_double;
      this_double = dynamic_cast<const Torch::DoubleTensor*>(this);
      res = this_double->dataR();
      break;
    case Torch::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

void* Torch::Tensor::dataW() {
  void* res = 0;
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      Torch::CharTensor* this_char;
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      res = this_char->dataW();
      break;
    case Torch::Tensor::Short:
      Torch::ShortTensor* this_short;
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      res = this_short->dataW();
      break;
    case Torch::Tensor::Int:
      Torch::IntTensor* this_int;
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      res = this_int->dataW();
      break;
    case Torch::Tensor::Long:
      Torch::LongTensor* this_long;
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      res = this_long->dataW();
      break;
    case Torch::Tensor::Float:
      Torch::FloatTensor* this_float;
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      res = this_float->dataW();
      break;
    case Torch::Tensor::Double:
      Torch::DoubleTensor* this_double;
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      res = this_double->dataW();
      break;
    case Torch::Tensor::Undefined:
    default:
      break;
  }
  return res;
}

void Torch::Tensor::resetFromData() {
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      Torch::CharTensor* this_char;
      this_char = dynamic_cast<Torch::CharTensor*>(this);
      this_char->resetFromData();
      break;
    case Torch::Tensor::Short:
      Torch::ShortTensor* this_short;
      this_short = dynamic_cast<Torch::ShortTensor*>(this);
      this_short->resetFromData();
      break;
    case Torch::Tensor::Int:
      Torch::IntTensor* this_int;
      this_int = dynamic_cast<Torch::IntTensor*>(this);
      this_int->resetFromData();
      break;
    case Torch::Tensor::Long:
      Torch::LongTensor* this_long;
      this_long = dynamic_cast<Torch::LongTensor*>(this);
      this_long->resetFromData();
      break;
    case Torch::Tensor::Float:
      Torch::FloatTensor* this_float;
      this_float = dynamic_cast<Torch::FloatTensor*>(this);
      this_float->resetFromData();
      break;
    case Torch::Tensor::Double:
      Torch::DoubleTensor* this_double;
      this_double = dynamic_cast<Torch::DoubleTensor*>(this);
      this_double->resetFromData();
      break;
    case Torch::Tensor::Undefined:
    default:
      break;
  }
}

long Torch::Tensor::stride(int dim) const {
  long res = 0;
  Torch::Tensor::Type type = m_datatype;

  switch(type)
  {
    case Torch::Tensor::Char:
      const Torch::CharTensor* this_char;
      this_char = dynamic_cast<const Torch::CharTensor*>(this);
      res = this_char->stride(dim);
      break;
    case Torch::Tensor::Short:
      const Torch::ShortTensor* this_short;
      this_short = dynamic_cast<const Torch::ShortTensor*>(this);
      res = this_short->stride(dim);
      break;
    case Torch::Tensor::Int:
      const Torch::IntTensor* this_int;
      this_int = dynamic_cast<const Torch::IntTensor*>(this);
      res = this_int->stride(dim);
      break;
    case Torch::Tensor::Long:
      const Torch::LongTensor* this_long;
      this_long = dynamic_cast<const Torch::LongTensor*>(this);
      res = this_long->stride(dim);
      break;
    case Torch::Tensor::Float:
      const Torch::FloatTensor* this_float;
      this_float = dynamic_cast<const Torch::FloatTensor*>(this);
      res = this_float->stride(dim);
      break;
    case Torch::Tensor::Double:
      const Torch::DoubleTensor* this_double;
      this_double = dynamic_cast<const Torch::DoubleTensor*>(this);
      res = this_double->stride(dim);
      break;
    case Torch::Tensor::Undefined:
    default:
      break;
  }
  return res;
}
