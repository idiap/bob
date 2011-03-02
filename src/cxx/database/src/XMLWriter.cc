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
#include "database/dataset_common.h"
#include "database/Exception.h"

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

      // Write author attribute if any
      if(dataset.getAuthor().compare("") )
        xmlNewProp( rootnode, (const xmlChar*)db::author, 
          (const xmlChar*)dataset.getAuthor().c_str() );

      // Write datetime attribute
      // if set use the existing date/time
      if(!dataset.getDateTime().is_not_a_date_time() )
        xmlNewProp( rootnode, (const xmlChar*)db::datetime, (const xmlChar*)
          (boost::posix_time::to_iso_extended_string(dataset.getDateTime())).c_str() );
      // otherwise use local date/time
      else
        xmlNewProp( rootnode, (const xmlChar*)db::datetime, (const xmlChar*)
          (boost::posix_time::to_iso_extended_string(boost::posix_time::second_clock::local_time())).c_str() );

      // Create PathList node if required
      PathList pl = dataset.getPathList();
      if( pl.paths().size() > 0 ) {
        xmlNodePtr pl_node = writePathList( doc, pl);
        if( pl.paths().size() > 0) // size after removing relative paths
          xmlAddChild( rootnode, pl_node );
      }

      // Create Arrayset nodes
      const std::map<size_t, boost::shared_ptr<Arrayset> >&
        arraysets = dataset.arraysetIndex(); 
      for(std::map<size_t, boost::shared_ptr<Arrayset> >::const_iterator 
          it=arraysets.begin(); it!=arraysets.end(); ++it)
      {
        xmlAddChild( rootnode, writeArrayset( doc, it->first, it->second, pl) );
      }
      // Create Relationset nodes
      const std::map<std::string, boost::shared_ptr<Relationset> >&
        relationsets = dataset.relationsetIndex(); 
      for(std::map<std::string, boost::shared_ptr<Relationset> >::const_iterator
          it=relationsets.begin(); it!=relationsets.end(); ++it)
      {
        xmlAddChild( rootnode, writeRelationset( doc, it->first, it->second) );
      }

      // Save the document to the specified XML file in UTF-8 format
      xmlSaveFormatFileEnc(filename, doc, "UTF-8", 1);

      // Free the memory
      xmlFreeDoc(doc);
    }


    xmlNodePtr XMLWriter::writeArrayset( xmlDocPtr doc, 
        size_t id, boost::shared_ptr<const Arrayset> a, 
        const PathList& pl, int precision, bool scientific) 
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
        (boost::lexical_cast<std::string>(id)).c_str() );

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
        // Reduce filename
        boost::filesystem::path red_filename = pl.reduce( a->getFilename() );
        // Write (reduced) file attribute
        xmlNewProp( arraysetnode, (const xmlChar*)db::file, (const xmlChar*)
          red_filename.string().c_str() );

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
          xmlAddChild( arraysetnode, writeArray( doc, *a_id, a->operator[](*a_id),
            pl, precision, scientific) );
        }
      }

      return arraysetnode;
    }


    xmlNodePtr XMLWriter::writeArray( xmlDocPtr doc, 
      size_t id, const Array a, const PathList& pl, int precision, bool scientific)
    {
      // Create the Arrayset node
      xmlNodePtr arraynode;
 
      // External Array
      if( a.getFilename().compare("") ) {
        arraynode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::external_array, 0);

        // Reduce filename
        boost::filesystem::path red_filename = pl.reduce( a.getFilename() );
        // Write (reduced) file attribute
        xmlNewProp( arraynode, (const xmlChar*)db::file, (const xmlChar*)
          red_filename.string().c_str() );

        // Write codec attribute
        std::string str( a.getCodec()->name() );
        xmlNewProp( arraynode, (const xmlChar*)db::codec, (const xmlChar*)
          str.c_str() );
      }
      // Inline Array
      else { 
        // Prepare the string stream
        std::ostringstream content;
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
        (boost::lexical_cast<std::string>(id)).c_str() );

      return arraynode;
    }


    xmlNodePtr XMLWriter::writeRelationset( xmlDocPtr doc, std::string name,
     boost::shared_ptr<const Relationset> r) 
    {
      // Create the Relationset node
      xmlNodePtr relationsetnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relationset, 0);

      // Write name attribute
      xmlNewProp( relationsetnode, (const xmlChar*)db::name, 
          (const xmlChar*)name.c_str() );

      // Add the Rule nodes to the relationset node
      const std::map<std::string, boost::shared_ptr<Rule> > rule = r->rules();
      for(std::map<std::string, boost::shared_ptr<Rule> >::const_iterator 
        it=rule.begin(); it!=rule.end(); ++it)
      {
        xmlAddChild( relationsetnode, writeRule( doc, it->first, it->second) );
      }

      // Add the Relation nodes to the relationset node
      const std::map<size_t, boost::shared_ptr<Relation> > 
        relation = r->relations();
      for(std::map<size_t, boost::shared_ptr<Relation> >::const_iterator 
        it=relation.begin(); it!=relation.end(); ++it)
      {
        xmlAddChild( relationsetnode, 
          writeRelation( doc, it->first, it->second) );
      }

      return relationsetnode;
    }
    

    xmlNodePtr XMLWriter::writeRule( xmlDocPtr doc, const std::string role,
      boost::shared_ptr<const Rule> r) 
    {
      // Create the Rule node
      xmlNodePtr rulenode = xmlNewDocNode(doc,0, (const xmlChar*)db::rule, 0);

      // Write arrayset-role attribute
      xmlNewProp( rulenode, (const xmlChar*)db::arrayset_role, 
          (const xmlChar*)role.c_str() );
      // Write min attribute
      xmlNewProp( rulenode, (const xmlChar*)db::min, (const xmlChar*)
        (boost::lexical_cast<std::string>(r->getMin())).c_str() );
      // Write max attribute
      xmlNewProp( rulenode, (const xmlChar*)db::max, (const xmlChar*)
        (boost::lexical_cast<std::string>(r->getMax())).c_str() );

      return rulenode;
    }

    
    xmlNodePtr XMLWriter::writeRelation( xmlDocPtr doc, size_t id, 
      boost::shared_ptr<const Relation> r) 
    {
      // Create the Relation node
      xmlNodePtr relationnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relation, 0);

      // Write id attribute
      xmlNewProp( relationnode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(id)).c_str() );

      // Add the Member nodes to the relation node
      const std::list<std::pair<size_t,size_t> > member = r->members();
      for(std::list<std::pair<size_t,size_t> >::const_iterator 
        it=member.begin(); it!=member.end(); ++it)
      {
        xmlAddChild( relationnode, writeMember( doc, it->first, it->second) );
      }
      return relationnode;
    }


    xmlNodePtr XMLWriter::writeMember( xmlDocPtr doc, 
      const size_t arrayset_id, const size_t array_id) 
    {
      // Create the Member node
      xmlNodePtr membernode; 
      if( array_id != 0)
        membernode = xmlNewDocNode(doc, 0, (const xmlChar*)db::member, 0);
      else
        membernode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset_member, 0);

      // Write arrayset-id attribute
      xmlNewProp( membernode, (const xmlChar*)db::arrayset_id, (const xmlChar*)
        (boost::lexical_cast<std::string>(arrayset_id)).c_str() );

      // Write array-id attribute if any
      if( array_id != 0)
        xmlNewProp( membernode, (const xmlChar*)db::array_id, (const xmlChar*)
          (boost::lexical_cast<std::string>(array_id)).c_str() );

      return membernode;
    }

    xmlNodePtr XMLWriter::writePathList( xmlDocPtr doc,
      db::PathList& pl)
    {
      // Create the PathList node
      xmlNodePtr pathlistnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::pathlist, 0);
      
      // Add the Entry nodes to the PathList node
      const std::list<boost::filesystem::path>& entries = pl.paths();
      std::list<boost::filesystem::path> to_remove;
      for(std::list<boost::filesystem::path>::const_iterator 
        it=entries.begin(); it!=entries.end(); ++it)
      {
        xmlNodePtr entrynode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::entry, 0);
        if( (*it).has_relative_path() )
          // Add in the list of paths to be removed
          to_remove.push_back( *it);
        else {
          // Add a path entry
          xmlNewProp( entrynode, (const xmlChar*)db::path, 
            (const xmlChar*)((*it).string().c_str()) );
          xmlAddChild( pathlistnode, entrynode );
        }
      }

      // Remove relative path
      for(std::list<boost::filesystem::path>::const_iterator 
        it=to_remove.begin(); it!=to_remove.end(); ++it)
        pl.remove( *it);

      return pathlistnode;
    }

  }}
}

