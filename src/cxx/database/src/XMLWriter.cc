/**
 * @file src/cxx/database/src/XMLWriter.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML writer for a dataset.
 */

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "database/XMLWriter.h"
#include "database/Dataset.h"
#include "database/Arrayset.h"
#include "database/Array.h"
#include "database/Relationset.h"

#include "core/logging.h"

namespace db = Torch::database;
namespace tc = Torch::core;
namespace tca = Torch::core::array;

namespace Torch {
  namespace database { namespace detail {

    XMLWriter::XMLWriter() { }

    XMLWriter::~XMLWriter() { }


    void XMLWriter::write(const char *filename, const db::Dataset& dataset)
    {
      xmlDocPtr doc;
      xmlNodePtr rootnode;

      doc = xmlNewDoc((const xmlChar*)"1.0");
      // Create the root node (Dataset) and set it in the document
      rootnode = xmlNewDocNode(doc, 0, (const xmlChar*)db::dataset, 0);
      xmlDocSetRootElement(doc, rootnode);

      // Write name attribute if any
      if(dataset.getName().compare("") )
        xmlNewProp( rootnode, (const xmlChar*)db::name, 
          (const xmlChar*)dataset.getName().c_str() );

      // Write version attribute if any
      if(dataset.getVersion() != 0)
        xmlNewProp( rootnode, (const xmlChar*)db::version, (const xmlChar*)
          (boost::lexical_cast<std::string>(dataset.getVersion())).c_str() );

      // Create Arrayset nodes
      const std::list<boost::shared_ptr<Arrayset> > arraysets = 
        dataset.arraysets();
      for(std::list<boost::shared_ptr<Arrayset> >::const_iterator 
        it=arraysets.begin(); it!=arraysets.end(); ++it)
      {
        xmlAddChild( rootnode, writeArrayset( doc, *it) );
      }
      // Create Relationset nodes
/*      for(Dataset::relationset_const_iterator it=dataset.relationset_begin(); 
        it!=dataset.relationset_end(); ++it)
      {
        xmlAddChild( rootnode, writeRelationset( doc, *it->second, b) );
      }
*/
      // Save the document to the specified XML file in UTF-8 format
      xmlSaveFormatFileEnc(filename, doc, "UTF-8", 1);
    }


    xmlNodePtr XMLWriter::writeArrayset( xmlDocPtr doc, 
      boost::shared_ptr<const Arrayset> a, 
      int precision, bool scientific) 
    {
      // Create the Arrayset node
      xmlNodePtr arraysetnode; 
      if( a->getFilename().compare("") )
        arraysetnode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::external_arrayset, 0);
      else
        arraysetnode = xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset, 0);

      // Write id attribute
      xmlNewProp( arraysetnode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(a->getId())).c_str() );

      // Write elementtype attribute
      std::string str;
      switch(a->getElementType()) {
        case tca::t_bool:
          str = db::t_bool; break;
        case tca::t_int8:
          str = db::t_int8; break;
        case tca::t_int16:
          str = db::t_int16; break;
        case tca::t_int32:
          str = db::t_int32; break;
        case tca::t_int64:
          str = db::t_int64; break;
        case tca::t_uint8:
          str = db::t_uint8; break;
        case tca::t_uint16:
          str = db::t_uint16; break;
        case tca::t_uint32:
          str = db::t_uint32; break;
        case tca::t_uint64:
          str = db::t_uint64; break;
        case tca::t_float32:
          str = db::t_float32; break;
        case tca::t_float64:
          str = db::t_float64; break;
        case tca::t_float128:
          str = db::t_float128; break;
        case tca::t_complex64:
          str = db::t_complex64; break;
        case tca::t_complex128:
          str = db::t_complex128; break;
        case tca::t_complex256:
          str = db::t_complex256; break;
        default:
          throw tc::Exception();
          break;
      }    
      xmlNewProp( arraysetnode, (const xmlChar*)db::elementtype, (const xmlChar*)
        str.c_str() );

      // Write shape attribute
      const size_t* shape = a->getShape();
      str = boost::lexical_cast<std::string>(shape[0]);
      for(size_t i=1; i<a->getNDim(); ++i)
        str += " " + boost::lexical_cast<std::string>(shape[i]);
      xmlNewProp( arraysetnode, (const xmlChar*)db::shape, (const xmlChar*)
        str.c_str() );

      // Write role attribute
      xmlNewProp( arraysetnode, (const xmlChar*)db::role, (const xmlChar*)
        a->getRole().c_str() );

      // Write file and loader attributes if any
      if( a->getFilename().compare("") )
      {
        // Write file attribute
        xmlNewProp( arraysetnode, (const xmlChar*)db::file, (const xmlChar*)
          a->getFilename().c_str() );

        // Write codec attribute
        str = a->getCodec()->name();
        // TODO: check that the codec exists in the registry?
        xmlNewProp( arraysetnode, (const xmlChar*)db::codec, (const xmlChar*)
          str.c_str() );
      }
      else {
        std::vector<size_t> ids;
        a->index( ids );
        // Create Array nodes
        for( std::vector<size_t>::const_iterator a_id=ids.begin(); 
          a_id!=ids.end(); ++a_id)
        {
          xmlAddChild( arraysetnode, writeArray( doc, a->operator[](*a_id),
            precision, scientific) );
        }
      }

      return arraysetnode;
    }


    xmlNodePtr XMLWriter::writeArray( xmlDocPtr doc, 
      const Array a, int precision, bool scientific)
    {
      // Create the Arrayset node
      xmlNodePtr arraynode;
 
      // External Array
      if( a.getFilename().compare("") ) {
        arraynode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::external_array, 0);

        // Write file attribute
        xmlNewProp( arraynode, (const xmlChar*)db::file, (const xmlChar*)
          a.getFilename().c_str() );

        // Write codec attribute
        std::string str( a.getCodec()->name() );
        xmlNewProp( arraynode, (const xmlChar*)db::codec, (const xmlChar*)
          str.c_str() );
      }
      // Inline Array
      else { 
        // Prepare the string stream
        std::stringstream content;
        content << std::setprecision(precision);
        if( scientific)
          content << std::scientific;

#define WRITE_ARRAY_DATA(T) switch(a.getNDim()) { \
  case 1: writeData<T,1>( content, a.get<T,1>() ); break; \
  case 2: writeData<T,2>( content, a.get<T,2>() ); break; \
  case 3: writeData<T,3>( content, a.get<T,3>() ); break; \
  case 4: writeData<T,4>( content, a.get<T,4>() ); break; \
  default: throw tc::Exception(); }

        switch( a.getElementType()) {
          case tca::t_bool:
            WRITE_ARRAY_DATA(bool);
            break;
          case tca::t_int8:
            WRITE_ARRAY_DATA(int8_t);
            break;
          case tca::t_int16:
            WRITE_ARRAY_DATA(int16_t);
            break;
          case tca::t_int32:
            WRITE_ARRAY_DATA(int32_t);
            break;
          case tca::t_int64:
            WRITE_ARRAY_DATA(int64_t);
            break;
          case tca::t_uint8:
            WRITE_ARRAY_DATA(uint8_t);
            break;
          case tca::t_uint16:
            WRITE_ARRAY_DATA(uint16_t);
            break;
          case tca::t_uint32:
            WRITE_ARRAY_DATA(uint32_t);
            break;
          case tca::t_uint64:
            WRITE_ARRAY_DATA(uint64_t);
            break;
          case tca::t_float32:
            WRITE_ARRAY_DATA(float);
            break;
          case tca::t_float64:
            WRITE_ARRAY_DATA(double);
            break;
          case tca::t_float128:
            WRITE_ARRAY_DATA(long double);
            break;
          case tca::t_complex64:
            WRITE_ARRAY_DATA(std::complex<float>);
            break;
          case tca::t_complex128:
            WRITE_ARRAY_DATA(std::complex<double>);
            break;
          case tca::t_complex256:
            WRITE_ARRAY_DATA(std::complex<long double>);
            break;
          default:
            throw tc::Exception();
            break;
        }

        TDEBUG3("Inline content: " << content.str());
        arraynode = xmlNewDocNode(doc, 0, (const xmlChar*)db::array, 
          (const xmlChar*)(content.str().c_str()));
      }

      // Write id attribute
      xmlNewProp( arraynode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(a.getId())).c_str() );

      return arraynode;
    }

/*
    xmlNodePtr XMLWriter::writeRelationset( xmlDocPtr doc, const Relationset& r,
      bool content_inline) 
    {
      // Create the Relationset node
      xmlNodePtr relationsetnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relationset, 0);

      // Write name attribute
      xmlNewProp( relationsetnode, (const xmlChar*)db::name, 
          (const xmlChar*)r.getName().c_str() );

      // Add the Rule nodes to the relationset node
      for(Relationset::rule_const_iterator it=r.rule_begin(); 
        it!=r.rule_end(); ++it)
      {
        xmlAddChild( relationsetnode, writeRule( doc, *it->second) );
      }

      // Add the Relation nodes to the relationset node
      for(Relationset::const_iterator it=r.begin(); it!=r.end(); ++it)
      {
        xmlAddChild( relationsetnode, writeRelation( doc, *it->second) );
      }

      return relationsetnode;
    }
    

    xmlNodePtr XMLWriter::writeRule( xmlDocPtr doc, const Rule& r) {
      // Create the Rule node
      xmlNodePtr rulenode = xmlNewDocNode(doc,0, (const xmlChar*)db::rule, 0);

      // Write arrayset-role attribute
      xmlNewProp( rulenode, (const xmlChar*)db::arrayset_role, 
          (const xmlChar*)r.getArraysetRole().c_str() );
      // Write min attribute
      xmlNewProp( rulenode, (const xmlChar*)db::min, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getMin())).c_str() );
      // Write max attribute
      xmlNewProp( rulenode, (const xmlChar*)db::max, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getMax())).c_str() );

      return rulenode;
    }
    
    xmlNodePtr XMLWriter::writeRelation( xmlDocPtr doc, const Relation& r) {
      // Create the Relation node
      xmlNodePtr relationnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relation, 0);

      // Write id attribute
      xmlNewProp( relationnode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getId())).c_str() );

      // Add the Member nodes to the relation node
      for(Relation::const_iterator it=r.begin(); it!=r.end(); ++it)
      {
        xmlAddChild( relationnode, writeMember( doc, *it->second) );
      }
      return relationnode;
    }

    xmlNodePtr XMLWriter::writeMember( xmlDocPtr doc, const Member& m) {
      // Create the Member node
      xmlNodePtr membernode; 
      if( m.getArrayId() != 0)
        membernode = xmlNewDocNode(doc, 0, (const xmlChar*)db::member, 0);
      else
        membernode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset_member, 0);

      // Write arrayset-id attribute
      xmlNewProp( membernode, (const xmlChar*)db::arrayset_id, (const xmlChar*)
        (boost::lexical_cast<std::string>(m.getArraysetId())).c_str() );

      // Write array-id attribute if any
      if( m.getArrayId() != 0)
        xmlNewProp( membernode, (const xmlChar*)db::array_id, (const xmlChar*)
          (boost::lexical_cast<std::string>(m.getArrayId())).c_str() );

      return membernode;
    }
*/

  }}
}

