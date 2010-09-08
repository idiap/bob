#ifndef TORCH5SPRO_TENSOR_WRAPPER_IMPL_H
#define TORCH5SPRO_TENSOR_WRAPPER_IMPL_H

namespace Torch 
{
  void Tensor::raiseError(std::string msg) const {
    std::cerr << "Error: " << msg << std::endl;
  }
  
  void Tensor::raiseFatalError(std::string msg) const {
    std::cerr << "Fatal Error: " << msg << std::endl;
    exit(-1);
  }
  
  void Tensor::setTensor( const Tensor *src) {
    Tensor::Type type = m_datatype;
    Tensor::Type src_type = src->getDatatype();
    // This seems to be reasonable (force the type to be specified before calling the function)
    if( src_type != type )
    {
      std::string msg("Tensor::setTensor() don't know how to set a Tensor from a different type. Try a copy instead.");
      raiseError(msg);
      return;
    }

    switch(src_type)
    {
      case Tensor::Char:
        const CharTensor* src_char;
        CharTensor* this_char;
        src_char = dynamic_cast<const CharTensor*>(src);
        this_char = dynamic_cast<CharTensor*>(this);
        this_char->setTensor( src_char);
        break;
      case Tensor::Short:
        const ShortTensor* src_short;
        ShortTensor* this_short;
        src_short = dynamic_cast<const ShortTensor*>(src);
        this_short = dynamic_cast<ShortTensor*>(this);
        this_short->setTensor( src_short);
        break;
      case Tensor::Int:
        const IntTensor* src_int;
        IntTensor* this_int;
        src_int = dynamic_cast<const IntTensor*>(src);
        this_int = dynamic_cast<IntTensor*>(this);
        this_int->setTensor( src_int);
        break;
      case Tensor::Long:
        const LongTensor* src_long;
        LongTensor* this_long;
        src_long = dynamic_cast<const LongTensor*>(src);
        this_long = dynamic_cast<LongTensor*>(this);
        this_long->setTensor( src_long);
        break;
      case Tensor::Float:
        const FloatTensor* src_float;
        FloatTensor* this_float;
        src_float = dynamic_cast<const FloatTensor*>(src);
        this_float = dynamic_cast<FloatTensor*>(this);
        this_float->setTensor( src_float);
        break;
      case Tensor::Double:
        const DoubleTensor* src_double;
        DoubleTensor* this_double;
        src_double = dynamic_cast<const DoubleTensor*>(src);
        this_double = dynamic_cast<DoubleTensor*>(this);
        this_double->setTensor( src_double);
        break;
      case Tensor::Undefined:
      default:
        std::string msg("Tensor::setTensor() don't know how to set a Tensor from a Undefined/Unknown type.");
        raiseError(msg);
        return;
    }
  }

  void Tensor::copy(const Tensor *src) {
    Tensor::Type type = m_datatype;
    Tensor::Type src_type = src->getDatatype();
    // This seems to be reasonable (force the type to be specified before calling the function)
    if( src_type == Tensor::Undefined || type == Tensor::Undefined )
    {
      std::cerr << "Error: Tensor::copy() don't know how to copy from or to an \"Undefined type\" Tensor." << std::endl;
      return;
    }

    switch(src_type)
    {
      case Tensor::Char:
        const CharTensor* src_char;
        src_char = dynamic_cast<const CharTensor*>(src);        
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_char);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_char);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_char);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_char);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_char);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_char);
            break;
          default:
            return;
        }
        break;

      case Tensor::Short:
        const ShortTensor* src_short;
        src_short = dynamic_cast<const ShortTensor*>(src);
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_short);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_short);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_short);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_short);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_short);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_short);
            break;
          default:
            return;
        }
        break;
      case Tensor::Int:
        const IntTensor* src_int;
        src_int = dynamic_cast<const IntTensor*>(src);
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_int);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_int);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_int);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_int);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_int);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_int);
            break;
          default:
            return;
        }
        break;
      case Tensor::Long:
        const LongTensor* src_long;
        src_long = dynamic_cast<const LongTensor*>(src);
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_long);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_long);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_long);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_long);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_long);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_long);
            break;
          default:
            return;
        }
        break;
      case Tensor::Float:
        const FloatTensor* src_float;
        src_float = dynamic_cast<const FloatTensor*>(src);
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_float);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_float);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_float);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_float);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_float);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_float);
            break;
          default:
            return;
        }
        break;
      case Tensor::Double:
        const DoubleTensor* src_double;
        src_double = dynamic_cast<const DoubleTensor*>(src);
        switch(type)
        {
          case Tensor::Char:
            CharTensor* this_char;
            this_char = dynamic_cast<CharTensor*>(this);
            this_char->copy( src_double);
            break;
          case Tensor::Short:
            ShortTensor* this_short;
            this_short = dynamic_cast<ShortTensor*>(this);
            this_short->copy( src_double);
            break;
          case Tensor::Int:
            IntTensor* this_int;
            this_int = dynamic_cast<IntTensor*>(this);
            this_int->copy( src_double);
            break;
          case Tensor::Long:
            LongTensor* this_long;
            this_long = dynamic_cast<LongTensor*>(this);
            this_long->copy( src_double);
            break;
          case Tensor::Float:
            FloatTensor* this_float;
            this_float = dynamic_cast<FloatTensor*>(this);
            this_float->copy( src_double);
            break;
          case Tensor::Double:
            DoubleTensor* this_double;
            this_double = dynamic_cast<DoubleTensor*>(this);
            this_double->copy( src_double);
            break;
          default:
            return;
        }
        break;
      case Tensor::Undefined:
        std::cerr << "Error: Tensor::copy() don't know how to set a Tensor from Undefined type." << std::endl;
      default:
        return;
    }
  }

  void Tensor::transpose( const Tensor *src, int dimension1, int dimension2) {
    Tensor::Type type = m_datatype;
    Tensor::Type src_type = src->getDatatype();
    // This seems to be reasonable (force the type to be specified before calling the function)
    if( src_type != type )
    {
      std::cerr << "Error: Tensor::transpose() don't know how to set a Tensor from a different type. Try a copy instead." << std::endl;
      return;
    }

    switch(src_type)
    {
      case Tensor::Char:
        const CharTensor* src_char;
        CharTensor* this_char;
        src_char = dynamic_cast<const CharTensor*>(src);
        this_char = dynamic_cast<CharTensor*>(this);
        this_char->transpose( src_char, dimension1, dimension2);
        break;
      case Tensor::Short:
        const ShortTensor* src_short;
        ShortTensor* this_short;
        src_short = dynamic_cast<const ShortTensor*>(src);
        this_short = dynamic_cast<ShortTensor*>(this);
        this_short->transpose( src_short, dimension1, dimension2);
        break;
      case Tensor::Int:
        const IntTensor* src_int;
        IntTensor* this_int;
        src_int = dynamic_cast<const IntTensor*>(src);
        this_int = dynamic_cast<IntTensor*>(this);
        this_int->transpose( src_int, dimension1, dimension2);
        break;
      case Tensor::Long:
        const LongTensor* src_long;
        LongTensor* this_long;
        src_long = dynamic_cast<const LongTensor*>(src);
        this_long = dynamic_cast<LongTensor*>(this);
        this_long->transpose( src_long, dimension1, dimension2);
        break;
      case Tensor::Float:
        const FloatTensor* src_float;
        FloatTensor* this_float;
        src_float = dynamic_cast<const FloatTensor*>(src);
        this_float = dynamic_cast<FloatTensor*>(this);
        this_float->transpose( src_float, dimension1, dimension2);
        break;
      case Tensor::Double:
        const DoubleTensor* src_double;
        DoubleTensor* this_double;
        src_double = dynamic_cast<const DoubleTensor*>(src);
        this_double = dynamic_cast<DoubleTensor*>(this);
        this_double->transpose( src_double, dimension1, dimension2);
        break;
      case Tensor::Undefined:
        std::cerr << "Error: Tensor::transpose() don't know how to set a Tensor from Undefined type." << std::endl;
      default:
        return;
    }
  }

  void Tensor::narrow (const Tensor *src, int dimension, long firstIndex,
      long size) {
    Tensor::Type type = m_datatype;
    Tensor::Type src_type = src->getDatatype();
    // This seems to be reasonable (force the type to be specified before calling the function)
    if( src_type != type )
    {
      std::cerr << "Error: Tensor::narrow() don't know how to set a Tensor from a different type. Try a copy instead." << std::endl;
      return;
    }

    switch(src_type)
    {
      case Tensor::Char:
        const CharTensor* src_char;
        CharTensor* this_char;
        src_char = dynamic_cast<const CharTensor*>(src);
        this_char = dynamic_cast<CharTensor*>(this);
        this_char->narrow( src_char, dimension, firstIndex, size);
        break;
      case Tensor::Short:
        const ShortTensor* src_short;
        ShortTensor* this_short;
        src_short = dynamic_cast<const ShortTensor*>(src);
        this_short = dynamic_cast<ShortTensor*>(this);
        this_short->narrow( src_short, dimension, firstIndex, size);
        break;
      case Tensor::Int:
        const IntTensor* src_int;
        IntTensor* this_int;
        src_int = dynamic_cast<const IntTensor*>(src);
        this_int = dynamic_cast<IntTensor*>(this);
        this_int->narrow( src_int, dimension, firstIndex, size);
        break;
      case Tensor::Long:
        const LongTensor* src_long;
        LongTensor* this_long;
        src_long = dynamic_cast<const LongTensor*>(src);
        this_long = dynamic_cast<LongTensor*>(this);
        this_long->narrow( src_long, dimension, firstIndex, size);
        break;
      case Tensor::Float:
        const FloatTensor* src_float;
        FloatTensor* this_float;
        src_float = dynamic_cast<const FloatTensor*>(src);
        this_float = dynamic_cast<FloatTensor*>(this);
        this_float->narrow( src_float, dimension, firstIndex, size);
        break;
      case Tensor::Double:
        const DoubleTensor* src_double;
        DoubleTensor* this_double;
        src_double = dynamic_cast<const DoubleTensor*>(src);
        this_double = dynamic_cast<DoubleTensor*>(this);
        this_double->narrow( src_double, dimension, firstIndex, size);
        break;
      case Tensor::Undefined:
        std::cerr << "Error: Tensor::narrow() don't know how to set a Tensor from Undefined type." << std::endl;
      default:
        return;
    }
  }

  void Tensor::select( const Tensor* src, int dimension, long sliceIndex) {
    Tensor::Type type = m_datatype;
    Tensor::Type src_type = src->getDatatype();
    // This seems to be reasonable (force the type to be specified before calling the function)
    if( src_type != type )
    {
      std::cerr << "Error: Tensor::select() don't know how to set a Tensor from a different type. Try a copy instead." << std::endl;
      return;
    }

    switch(src_type)
    {
      case Tensor::Char:
        const CharTensor* src_char;
        CharTensor* this_char;
        src_char = dynamic_cast<const CharTensor*>(src);
        this_char = dynamic_cast<CharTensor*>(this);
        this_char->select( src_char, dimension, sliceIndex);
        break;
      case Tensor::Short:
        const ShortTensor* src_short;
        ShortTensor* this_short;
        src_short = dynamic_cast<const ShortTensor*>(src);
        this_short = dynamic_cast<ShortTensor*>(this);
        this_short->select( src_short, dimension, sliceIndex);
        break;
      case Tensor::Int:
        const IntTensor* src_int;
        IntTensor* this_int;
        src_int = dynamic_cast<const IntTensor*>(src);
        this_int = dynamic_cast<IntTensor*>(this);
        this_int->select( src_int, dimension, sliceIndex);
        break;
      case Tensor::Long:
        const LongTensor* src_long;
        LongTensor* this_long;
        src_long = dynamic_cast<const LongTensor*>(src);
        this_long = dynamic_cast<LongTensor*>(this);
        this_long->select( src_long, dimension, sliceIndex);
        break;
      case Tensor::Float:
        const FloatTensor* src_float;
        FloatTensor* this_float;
        src_float = dynamic_cast<const FloatTensor*>(src);
        this_float = dynamic_cast<FloatTensor*>(this);
        this_float->select( src_float, dimension, sliceIndex);
        break;
      case Tensor::Double:
        const DoubleTensor* src_double;
        DoubleTensor* this_double;
        src_double = dynamic_cast<const DoubleTensor*>(src);
        this_double = dynamic_cast<DoubleTensor*>(this);
        this_double->select( src_double, dimension, sliceIndex);
        break;
      case Tensor::Undefined:
        std::cerr << "Error: Tensor::select() don't know how to set a Tensor from Undefined type." << std::endl;
      default:
        return;
    }
  }

  Tensor* Tensor::select( int dimension, long sliceIndex) const {
    Tensor* res = 0;
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        const CharTensor* this_char;
        this_char = dynamic_cast<const CharTensor*>(this);
        res = this_char->select( dimension, sliceIndex);
        break;
      case Tensor::Short:
        const ShortTensor* this_short;
        this_short = dynamic_cast<const ShortTensor*>(this);
        res = this_short->select( dimension, sliceIndex);
        break;
      case Tensor::Int:
        const IntTensor* this_int;
        this_int = dynamic_cast<const IntTensor*>(this);
        res = this_int->select( dimension, sliceIndex);
        break;
      case Tensor::Long:
        const LongTensor* this_long;
        this_long = dynamic_cast<const LongTensor*>(this);
        res = this_long->select( dimension, sliceIndex);
        break;
      case Tensor::Float:
        const FloatTensor* this_float;
        this_float = dynamic_cast<const FloatTensor*>(this);
        res = this_float->select( dimension, sliceIndex);
        break;
      case Tensor::Double:
        const DoubleTensor* this_double;
        this_double = dynamic_cast<const DoubleTensor*>(this);
        res = this_double->select( dimension, sliceIndex);
        break;
      case Tensor::Undefined:
        std::cerr << "Error: Tensor::select() don't know how to set a Tensor from Undefined type." << std::endl;
      default:
        break;
    }
    return res;
  }

  int Tensor::typeSize() const {
    int res = 0;
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        res = sizeof(char);
        break;
      case Tensor::Short:
        res = sizeof(short);
        break;
      case Tensor::Int:
        res = sizeof(int);
        break;
      case Tensor::Long:
        res = sizeof(long);
        break;
      case Tensor::Float:
        res = sizeof(float);
        break;
      case Tensor::Double:
        res = sizeof(double);
        break;
      case Tensor::Undefined:
      default:
        break;
    }
    return res;
  }

  const void* Tensor::dataR() const {
    const void* res = 0;
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        const CharTensor* this_char;
        this_char = dynamic_cast<const CharTensor*>(this);
        res = this_char->dataR();
        break;
      case Tensor::Short:
        const ShortTensor* this_short;
        this_short = dynamic_cast<const ShortTensor*>(this);
        res = this_short->dataR();
        break;
      case Tensor::Int:
        const IntTensor* this_int;
        this_int = dynamic_cast<const IntTensor*>(this);
        res = this_int->dataR();
        break;
      case Tensor::Long:
        const LongTensor* this_long;
        this_long = dynamic_cast<const LongTensor*>(this);
        res = this_long->dataR();
        break;
      case Tensor::Float:
        const FloatTensor* this_float;
        this_float = dynamic_cast<const FloatTensor*>(this);
        res = this_float->dataR();
        break;
      case Tensor::Double:
        const DoubleTensor* this_double;
        this_double = dynamic_cast<const DoubleTensor*>(this);
        res = this_double->dataR();
        break;
      case Tensor::Undefined:
      default:
        break;
    }
    return res;
  }

  void* Tensor::dataW() {
    void* res = 0;
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        CharTensor* this_char;
        this_char = dynamic_cast<CharTensor*>(this);
        res = this_char->dataW();
        break;
      case Tensor::Short:
        ShortTensor* this_short;
        this_short = dynamic_cast<ShortTensor*>(this);
        res = this_short->dataW();
        break;
      case Tensor::Int:
        IntTensor* this_int;
        this_int = dynamic_cast<IntTensor*>(this);
        res = this_int->dataW();
        break;
      case Tensor::Long:
        LongTensor* this_long;
        this_long = dynamic_cast<LongTensor*>(this);
        res = this_long->dataW();
        break;
      case Tensor::Float:
        FloatTensor* this_float;
        this_float = dynamic_cast<FloatTensor*>(this);
        res = this_float->dataW();
        break;
      case Tensor::Double:
        DoubleTensor* this_double;
        this_double = dynamic_cast<DoubleTensor*>(this);
        res = this_double->dataW();
        break;
      case Tensor::Undefined:
      default:
        break;
    }
    return res;
  }

  void Tensor::resetFromData() {
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        CharTensor* this_char;
        this_char = dynamic_cast<CharTensor*>(this);
        this_char->resetFromData();
        break;
      case Tensor::Short:
        ShortTensor* this_short;
        this_short = dynamic_cast<ShortTensor*>(this);
        this_short->resetFromData();
        break;
      case Tensor::Int:
        IntTensor* this_int;
        this_int = dynamic_cast<IntTensor*>(this);
        this_int->resetFromData();
        break;
      case Tensor::Long:
        LongTensor* this_long;
        this_long = dynamic_cast<LongTensor*>(this);
        this_long->resetFromData();
        break;
      case Tensor::Float:
        FloatTensor* this_float;
        this_float = dynamic_cast<FloatTensor*>(this);
        this_float->resetFromData();
        break;
      case Tensor::Double:
        DoubleTensor* this_double;
        this_double = dynamic_cast<DoubleTensor*>(this);
        this_double->resetFromData();
        break;
      case Tensor::Undefined:
      default:
        break;
    }
  }

  long Tensor::stride(int dim) const {
    long res = 0;
    Tensor::Type type = m_datatype;

    switch(type)
    {
      case Tensor::Char:
        const CharTensor* this_char;
        this_char = dynamic_cast<const CharTensor*>(this);
        res = this_char->stride(dim);
        break;
      case Tensor::Short:
        const ShortTensor* this_short;
        this_short = dynamic_cast<const ShortTensor*>(this);
        res = this_short->stride(dim);
        break;
      case Tensor::Int:
        const IntTensor* this_int;
        this_int = dynamic_cast<const IntTensor*>(this);
        res = this_int->stride(dim);
        break;
      case Tensor::Long:
        const LongTensor* this_long;
        this_long = dynamic_cast<const LongTensor*>(this);
        res = this_long->stride(dim);
        break;
      case Tensor::Float:
        const FloatTensor* this_float;
        this_float = dynamic_cast<const FloatTensor*>(this);
        res = this_float->stride(dim);
        break;
      case Tensor::Double:
        const DoubleTensor* this_double;
        this_double = dynamic_cast<const DoubleTensor*>(this);
        res = this_double->stride(dim);
        break;
      case Tensor::Undefined:
      default:
        break;
    }
    return res;
  }

}

#endif
