/**
 * @file src/cxx/database/src/XMLParser.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML parser for a dataset.
 */

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <libxml/xmlschemas.h>

#include "database/XMLParser.h"
#include "database/Dataset.h"
#include "database/Exception.h"
#include "database/dataset_common.h"

namespace db = Torch::database;
namespace tdd = Torch::database::detail;
namespace tca = Torch::core::array;

namespace fs = boost::filesystem;

/**
 * Removes the last component from the path, supposing it is complete. If it is
 * only root_path(), just return it.
 */
static fs::path trim_one(const fs::path& p) {
  if (p == p.root_path()) return p;

  fs::path retval;
  for (fs::path::iterator it = p.begin(); it!=p.end(); ++it) {
    fs::path::iterator next = it; 
    ++next; //< for the lack of better support in boost::filesystem V2
    if (next == p.end()) break; //< == skip the last bit
    retval /= *it;
  }
  return retval;
}

namespace Torch { namespace database { namespace detail {


  XMLParser::XMLParser() { }

  XMLParser::~XMLParser() { }

  void XMLParser::validateXMLSchema(xmlDocPtr doc) {
    // Get path to the XML Schema definition
    char *schema_path = getenv("TORCH_SCHEMA_PATH");
    if( !schema_path || !strcmp( schema_path, "") ) {
      Torch::core::error << "Environment variable $TORCH_SCHEMA_PATH " << 
        "is not set. " << "Have you setup your working environment " << 
        "correctly?" << std::endl;
      throw XMLException();
    }
    char schema_full_path[1024];
    strcpy( schema_full_path, schema_path);
    strcat( schema_full_path, "/dataset.xsd" );

    // Load the XML schema from the file
    xmlDocPtr xsd_doc = xmlReadFile(schema_full_path, 0, 0);
    if(xsd_doc == 0) {
      Torch::core::error << "Unable to load the XML schema" << std::endl;
      throw XMLException();        
    }
    // Create an XML schema parse context
    xmlSchemaParserCtxtPtr xsd_parser = xmlSchemaNewDocParserCtxt(xsd_doc);
    if(xsd_parser == 0) {
      xmlFreeDoc(xsd_doc);
      Torch::core::error << "Unable to create the XML schema parse " << 
        "context." << std::endl;
      throw XMLException();        
    }
    // Parse the XML schema definition and check its correctness
    xmlSchemaPtr xsd_schema = xmlSchemaParse(xsd_parser);
    if(xsd_schema == 0) {
      xmlSchemaFreeParserCtxt(xsd_parser);
      xmlFreeDoc(xsd_doc); 
      Torch::core::error << "Invalid XML Schema definition." << std::endl;
      throw XMLException();        
    }
    // Create an XML schema validation context for the schema
    xmlSchemaValidCtxtPtr xsd_valid = xmlSchemaNewValidCtxt(xsd_schema);
    if(xsd_valid == 0) {
      xmlSchemaFree(xsd_schema);
      xmlSchemaFreeParserCtxt(xsd_parser);
      xmlFreeDoc(xsd_doc);
      Torch::core::error << "Unable to create an XML Schema validation " << 
        "context." <<  std::endl;
      throw XMLException();
    }

    // Check that the document is valid wrt. to the schema, and throw an 
    // exception otherwise.
    if(xmlSchemaValidateDoc(xsd_valid, doc)) {
      Torch::core::error << "The XML file is NOT valid with respect " << 
        "to the XML schema." << std::endl;
      throw XMLException();
    }

    xmlSchemaFreeValidCtxt(xsd_valid);
    xmlSchemaFree(xsd_schema);
    xmlSchemaFreeParserCtxt(xsd_parser);
    xmlFreeDoc(xsd_doc);
  }


  void XMLParser::load(const char* filename, db::Dataset& dataset, 
      size_t check_level) 
  {
    // Prepare the XML parser
    xmlInitParser();    

    // Parse the XML file with libxml2
    xmlDocPtr doc = xmlParseFile(filename);
    xmlNodePtr cur; 
    xmlNodePtr cur_svg; 

    // Check validity of the XML file
    if(doc == 0 ) {
      Torch::core::error << "Document " << filename << 
        " was not parsed successfully." << std::endl;
      throw XMLException();
    }

    // Check that the XML file is not empty
    cur = xmlDocGetRootElement(doc);
    if(cur == 0) { 
      Torch::core::error << "Document " << filename << " is empty." << 
        std::endl;
      xmlFreeDoc(doc);
      throw XMLException();
    }

    // Check that the XML file contains a dataset
    if( strcmp((const char*)cur->name, db::dataset) ) {
      Torch::core::error << "Document " << filename << 
        " is of the wrong type (!= dataset)." << std::endl;
      xmlFreeDoc(doc);
      throw XMLException();
    } 

    // Validate the XML document against the XML Schema
    // Throw an exception in case of failure
    validateXMLSchema( doc);

    // Parse Dataset Attributes
    // 1/ Parse name
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::name);
    dataset.setName( ( (str!=0?(const char *)str:"") ) );
    TDEBUG3("Name: " << dataset.getName());
    xmlFree(str);

    // 2/ Parse version 
    str = xmlGetProp(cur, (const xmlChar*)db::version);
    dataset.setVersion( str!=0 ? 
        boost::lexical_cast<size_t>((const char *)str) : 0 );
    TDEBUG3("Version: " << dataset.getVersion());
    xmlFree(str);

    // 3/ Parse author
    str = xmlGetProp(cur, (const xmlChar*)db::author);
    dataset.setAuthor( ( (str!=0?(const char *)str:"") ) );
    TDEBUG3("Author: " << dataset.getAuthor());
    xmlFree(str);

    // 4/ Parse date
    str = xmlGetProp(cur, (const xmlChar*)db::datetime);
    std::string str_iso_extended( str!=0 ? (const char *)str : "" );
    // Make an iso_extended string
    str_iso_extended[10] = ' ';
    dataset.setDateTime( str!=0 ? 
      boost::posix_time::ptime(boost::posix_time::time_from_string(
        str_iso_extended)) :
      boost::posix_time::ptime() );
    TDEBUG3("DateTime: " << boost::posix_time::to_iso_extended_string(dataset.getDateTime()));
    xmlFree(str);

    // 5/ Create an empty PathList and parse the PathList if any
    // Parse the PathList in the XML file if any
    db::PathList pl;
    db::PathList pl_tmp(".");
    boost::filesystem::path full_path = pl_tmp.locate( filename );
    pl.setCurrentPath( trim_one(full_path) );
    cur = cur_svg = cur->xmlChildrenNode;
    while(cur != 0) { 
      if( !strcmp((const char*)cur->name, db::pathlist) ) {
        parsePathList(cur, pl);
        break; // At most 1 PathList
      }
      else if( !strcmp((const char*)cur->name, db::arrayset) || 
          !strcmp((const char*)cur->name, db::external_arrayset) ) 
        break; // PathList should be defined before Arraysets/Arrays
      cur = cur->next;
    }
    // Attach the pathlist to the Dataset
    dataset.setPathList( pl );

    // Parse Arraysets and Relationsets
    cur = cur_svg;
    while(cur != 0) { 
      // Parse an arrayset and add it to the dataset
      if( !strcmp((const char*)cur->name, db::arrayset) || 
          !strcmp((const char*)cur->name, db::external_arrayset) ) {
        std::pair<size_t, boost::shared_ptr<db::Arrayset> >
          pcur = parseArrayset(cur, pl);
        dataset.add(pcur.first, pcur.second);
      }
      // Parse a relationset and add it to the dataset
      else if( !strcmp((const char*)cur->name, db::relationset) ) {
        std::pair<std::string, boost::shared_ptr<db::Relationset> >
          pcur = parseRelationset(cur, dataset);
        dataset.add(pcur.first, pcur.second);
      }
      cur = cur->next;
    }

    

    // High-level checks (which can not be done by libxml2)
    /*    if( check_level>= 1)
          {
          TDEBUG3(std::endl << "HIGH-LEVEL CHECKS");
    // Iterate over the relationsets
    for( Dataset::relationset_const_iterator 
    relationset = dataset.relationset_begin();
    relationset != dataset.relationset_end(); ++relationset )
    {
    TDEBUG3("Relationset name: " << relationset->second->getName()); 

    // Check that the rules are correct.
    //   (arrayset-role refers to an existing string role)
    for( Relationset::rule_const_iterator 
    rule = relationset->second->rule_begin();
    rule != relationset->second->rule_end(); ++rule )
    {
    TDEBUG3("Rule role: " << rule->second->getArraysetRole()); 
    bool found = false;
    for( Dataset::const_iterator arrayset = dataset.begin(); 
    arrayset != dataset.end(); ++arrayset )
    {
    if( !rule->second->getArraysetRole().compare( 
    (*arrayset)->getRole() ) )
    {
    found = true;
    break;
    }
    }

    if( !found ) {
    error << "Rule refers to a non-existing arrayset-role (" << 
    rule->second->getArraysetRole() << ")." << std::endl;
    throw Exception();
    }
    }

    // Check that the relations are correct
    for( Relationset::const_iterator 
    relation = relationset->second->begin();
    relation != relationset->second->end(); ++relation )
    {
    TDEBUG3("Relation id: " << relation->second->getId()); 

    // Check that for each rule in the relationset, the multiplicity
    // of the members is correct.
    for( Relationset::rule_const_iterator 
    rule = relationset->second->rule_begin();
    rule != relationset->second->rule_end(); ++rule )
    {
    TDEBUG3("Rule id: " << rule->second->getArraysetRole());

    size_t counter = 0;
    bool check_ok = true;
    for( Relation::const_iterator member = relation->second->begin();
    member != relation->second->end(); ++member )
    {
    TDEBUG3("  Member ids: " << member->second->getArrayId()
    << "," << member->second->getArraysetId());
    TDEBUG3("  " << (*m_id_role)[member->second->getArraysetId()]);
    TDEBUG3("  " << rule->second->getArraysetRole() << std::endl);

    if( !(*m_id_role)[member->second->getArraysetId()].compare( 
    rule->second->getArraysetRole() ) )
    {
    TDEBUG3("  Array id: " << member->second->getArrayId());
    if( member->second->getArrayId()!=0 )
    ++counter;
    else // Arrayset-member
    {
    const Arrayset &ar = 
    dataset[member->second->getArraysetId()];
    if( ar.getIsLoaded() )
      counter += ar.getNArrays();
    else if( check_level >= 2 ) {
      ;//TODO: load the arrayset
      // counter += ar.getNArrays();
    }
    else
      check_ok = false;
  }
  }
  }

  TDEBUG3("  Counter: " << counter);
  if( check_ok && ( counter<rule->second->getMin() || 
        (rule->second->getMax()!=0 && counter>rule->second->getMax()) ) )
  {
    error << "Relation (id=" << relation->second->getId() << 
      ") is not valid." << std::endl;
    throw Exception();
  }
  else if(!check_ok)
    warn << "Relation (id=" << relation->second->getId() <<
      ") has not been fully checked, because of external data." << 
      std::endl;
  }

  // Check that there is no member referring to a non-existing rule.
  for( Relation::const_iterator member = relation->second->begin();
      member != relation->second->end(); ++member )
  {
    TDEBUG3("  Member ids: " << member->second->getArrayId() <<
        "," << member->second->getArraysetId());

    bool found = false;
    for( Relationset::rule_const_iterator 
        rule = relationset->second->rule_begin();
        rule != relationset->second->rule_end(); ++rule )
    {
      TDEBUG3("Rule id: " << rule->second->getArraysetRole()); 
      if( !(*m_id_role)[member->second->getArraysetId()].compare(
            rule->second->getArraysetRole() ) )
      {
        found = true;
        break;
      }
    }

    if( !found ) {
      error << "Member (id:" << member->second->getArrayId() <<
        "," << member->second->getArraysetId() << 
        ") refers to a non-existing rule." << std::endl;
      throw Exception();
    }
  }
  }
  }
  }
  */

    // Deallocates memory
    xmlFreeDoc(doc);
    xmlCleanupParser();    
  }


  std::pair<std::string, boost::shared_ptr<db::Relationset> >
  XMLParser::parseRelationset(const xmlNodePtr cur, const db::Dataset& d)
  {
    boost::shared_ptr<Relationset> relationset(new Relationset());
    // Set parent dataset
    relationset->setParent(&d);

    // Parse name
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::name);
    std::string str_name(str!=0 ? (const char *)str : "");
    TDEBUG3("Name: " << str_name);
    xmlFree(str);

    // Parse the relations and rules
    xmlNodePtr cur_relationrule = cur->xmlChildrenNode;
    while(cur_relationrule != 0) { 
      // Parse a relation and add it to the relationset
      if( !strcmp((const char*)cur_relationrule->name, db::relation) ) {
        std::pair<size_t, boost::shared_ptr<db::Relation> > 
          pcur = parseRelation( cur_relationrule);
        relationset->add( pcur.first, pcur.second );
      }
      // Parse a rule and add it to the relationset
      else if( !strcmp((const char*)cur_relationrule->name, db::rule) ) {
        std::pair<std::string, boost::shared_ptr<db::Rule> > 
          pcur = parseRule( cur_relationrule);
        relationset->add( pcur.first, pcur.second );
      }
      cur_relationrule = cur_relationrule->next;
    }

    // Return the arrayset
    return std::make_pair(str_name, relationset);
  }


  std::pair<std::string, boost::shared_ptr<db::Rule> >
  XMLParser::parseRule(const xmlNodePtr cur) 
  {
    // Parse arrayset-role
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::arrayset_role);
    std::string str_ArraysetRole( str!=0 ? (const char *)str : "");
    TDEBUG3("  Arrayset-role: " << str_ArraysetRole);
    xmlFree(str);

    // Parse min
    str = xmlGetProp(cur, (const xmlChar*)db::min);
    size_t r_min = (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("  Min: " << r_min);
    xmlFree(str);

    // Parse max
    str = xmlGetProp(cur, (const xmlChar*)db::max);
    size_t r_max = (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("  Max: " << r_max);
    xmlFree(str);

    // Create the Rule
    boost::shared_ptr<Rule> rule(new Rule(r_min, r_max));

    return std::make_pair(str_ArraysetRole, rule);
  }


  std::pair<size_t, boost::shared_ptr<db::Relation> >
  XMLParser::parseRelation(const xmlNodePtr cur) 
  {
    // Parse id
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::id);
    size_t r_id = (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("  Id: " << r_id);
    xmlFree(str);

    // Create the relation
    boost::shared_ptr<Relation> relation(new Relation());

    // Parse the members and add them to the relation
    xmlNodePtr cur_member = cur->xmlChildrenNode;
    while(cur_member != 0) { 
      // Parse a member and add it to the relation
      if( !strcmp((const char*)cur_member->name, db::member) ) { 
        std::pair<size_t,size_t> p_cur = parseMember(cur_member);
        relation->add( p_cur.first, p_cur.second);
      }
      else if( !strcmp((const char*)cur_member->name, db::arrayset_member) )
        relation->add( parseArraysetMember(cur_member) );
      cur_member = cur_member->next;
    }

    return std::make_pair(r_id, relation);
  }


  std::pair<size_t,size_t> XMLParser::parseMember(const xmlNodePtr cur) 
  {
    // Parse array-id
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::array_id);
    size_t array_id = 
      (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("    Array-id: " << array_id);
    xmlFree(str);

    // Parse arrayset-id
    str = xmlGetProp(cur, (const xmlChar*)db::arrayset_id);
    size_t arrayset_id = 
      (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("    Arrayset-id: " << arrayset_id);
    xmlFree(str);

    return std::make_pair(arrayset_id, array_id);
  } 

  size_t XMLParser::parseArraysetMember(const xmlNodePtr cur) 
  {
    // Parse arrayset-id
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::arrayset_id);
    size_t arrayset_id = 
      (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("    Arrayset-id: " << arrayset_id);
    xmlFree(str);

    return arrayset_id;
  } 

  std::pair<size_t, boost::shared_ptr<db::Arrayset> >
  XMLParser::parseArrayset( const xmlNodePtr cur, const PathList& pl)
  {
    // Parse filename
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::file);
    std::string str_filename( (str!=0?(const char*)str:"") );
    TDEBUG3("File: " << str_filename);
    xmlFree(str);

    // Parse codec
    str = xmlGetProp(cur, (const xmlChar*)db::codec);
    std::string str_codecname( str!=0 ? (const char*)str: "" );
    TDEBUG3("Codec: " << str_codecname);
    xmlFree(str);

    // Create a smart pointer which will contain the Arrayset to be returned.
    boost::shared_ptr<db::Arrayset> arrayset;

    // Inline arrayset
    if( !str_filename.compare("") )
    {
      // Parse elementtype
      str = xmlGetProp(cur, (const xmlChar*)db::elementtype);
      if( str==0 ) {
        Torch::core::error << "Elementtype is not specified in arrayset (id: " << 
          id << ")." << std::endl;
        throw XMLException();
      }
      tca::ElementType elem_type;
      std::string str_element_type( (const char*)str );
      if( !str_element_type.compare( db::t_bool ) )
        elem_type = tca::t_bool;
      else if( !str_element_type.compare( db::t_uint8 ) )
        elem_type = tca::t_uint8;
      else if( !str_element_type.compare( db::t_uint16 ) )
        elem_type = tca::t_uint16;
      else if( !str_element_type.compare( db::t_uint32 ) )
        elem_type = tca::t_uint32;
      else if( !str_element_type.compare( db::t_uint64 ) )
        elem_type = tca::t_uint64;
      else if( !str_element_type.compare( db::t_int8 ) )
        elem_type = tca::t_int8;
      else if( !str_element_type.compare( db::t_int16 ) )
        elem_type = tca::t_int16;
      else if( !str_element_type.compare( db::t_int32 ) )
        elem_type = tca::t_int32;
      else if( !str_element_type.compare( db::t_int64 ) )
        elem_type = tca::t_int64;
      else if( !str_element_type.compare( db::t_float32 ) )
        elem_type = tca::t_float32;
      else if( !str_element_type.compare( db::t_float64 ) )
        elem_type = tca::t_float64;
      else if( !str_element_type.compare( db::t_float128 ) )
        elem_type = tca::t_float128;
      else if( !str_element_type.compare( db::t_complex64 ) )
        elem_type = tca::t_complex64;
      else if( !str_element_type.compare( db::t_complex128 ) )
        elem_type = tca::t_complex128;
      else if( !str_element_type.compare( db::t_complex256 ) )
        elem_type = tca::t_complex256;
      else
        elem_type = tca::t_unknown;
      TDEBUG3("Elementtype: " << elem_type);
      xmlFree(str);

      // Parse shape
      size_t shape[tca::N_MAX_DIMENSIONS_ARRAY];
      for(size_t i=0; i<tca::N_MAX_DIMENSIONS_ARRAY; ++i)
        shape[i]=0;
      str = xmlGetProp(cur, (const xmlChar*)db::shape);
      if( str==0 ) {
        Torch::core::error << "Elementtype is not specified in arrayset (id: " << 
          id << ")." << std::endl;
        throw XMLException();
      }
      // Tokenize the shape string to extract the dimensions
      std::string str_shape((const char *)str);
      boost::tokenizer<> tok(str_shape);
      size_t count=0;
      for( boost::tokenizer<>::iterator it=tok.begin(); it!=tok.end(); 
          ++it, ++count ) 
      {
        if(count>=tca::N_MAX_DIMENSIONS_ARRAY) {
          Torch::core::error << "Shape is not valid in arrayset (id: " << 
            id << "). Maximum number of dimensions is " <<
            tca::N_MAX_DIMENSIONS_ARRAY << "." << std::endl;
          throw XMLException();        
        }
        shape[count] = boost::lexical_cast<size_t>((*it).c_str());
      }
      TDEBUG3("Nb dimensions: " << count);
      TDEBUG3("Shape: (" << shape[0] << "," << shape[1] << ","<< 
          shape[2] << "," << shape[3] << ")");
      xmlFree(str);

      // Create a vector which will contain the database arrays (keep insertion
      // order)
      std::vector<size_t> array_ids;
      std::vector<boost::shared_ptr<db::Array> > arrays;

      // Parse the data
      xmlNodePtr cur_data = cur->xmlChildrenNode;

      while (cur_data != 0) { 
        // Process an array
        if ( !strcmp( (const char*)cur_data->name, db::array) || 
            !strcmp( (const char*)cur_data->name, db::external_array) ) {
          std::pair<size_t, boost::shared_ptr<db::Array> > 
            pcur = parseArray( cur_data, elem_type, shape, count, pl);
          array_ids.push_back(pcur.first);
          arrays.push_back(pcur.second);
        }
        cur_data = cur_data->next;
      }
    
      // Create an inline arrayset from the vector
      arrayset.reset(new db::Arrayset());
      for (size_t i=0; i<arrays.size(); ++i) 
        arrayset->add(array_ids[i], arrays[i]);

    }
    else // External Arrayset 
      arrayset.reset( 
        new db::Arrayset(pl.locate(str_filename).string(), str_codecname) );

    // Parse id
    str = xmlGetProp(cur, (const xmlChar*)db::id);
    size_t id = (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("Id: " << id);
    xmlFree(str);

    // Parse role
    str = xmlGetProp(cur, (const xmlChar*)db::role);
    std::string str_role(str!=0?(const char *)str:"");
    TDEBUG3("Role: " << str_role);
    xmlFree(str);
    arrayset->setRole( str_role );

    // Return the arrayset
    return std::make_pair(id, arrayset);
  }


  std::pair<size_t, boost::shared_ptr<db::Array> > 
  XMLParser::parseArray( const xmlNodePtr cur, 
    tca::ElementType elem, size_t shape[tca::N_MAX_DIMENSIONS_ARRAY], 
    size_t nb_dim, const PathList& pl)
  {
    // Parse codec
    xmlChar *str;
    str = xmlGetProp(cur, (const xmlChar*)db::codec);
    std::string str_codecname( str!=0 ? (const char*)str: "" );
    TDEBUG3("  Array Codec: " << str_codecname);
    xmlFree(str);

    // Parse filename
    str = xmlGetProp(cur, (const xmlChar*)db::file);
    std::string str_filename(str!=0?(const char*)str:"");
    TDEBUG3("  Array File: " << str_filename);
    xmlFree(str);

    // Create the array
    boost::shared_ptr<db::Array> array;

    // Parse the data contained in the XML file
    if( !str_filename.compare("") )
    {
      // Preliminary for the processing of the content of the array
      xmlChar* content = xmlNodeGetContent(cur);
      std::string data( (const char *)content);
      boost::char_separator<char> sep(" ;|");
      boost::tokenizer<boost::char_separator<char> > tok(data, sep);
      xmlFree(content);

      // Switch over the possible type
      blitz::TinyVector<int,1> b_shape1;
      blitz::TinyVector<int,2> b_shape2;
      blitz::TinyVector<int,3> b_shape3;
      blitz::TinyVector<int,4> b_shape4;
#define PARSE_ARRAY_DATA(T) switch(nb_dim) { \
  case 1: { b_shape1(0) = shape[0]; \
    blitz::Array<T,1> bla = parseArrayData<T,1>( tok, b_shape1 ); \
    array.reset( new db::Array( bla )); } break; \
  case 2: { b_shape2(0) = shape[0]; \
    b_shape2(1) = shape[1]; \
    blitz::Array<T,2> bla = parseArrayData<T,2>( tok, b_shape2 ); \
    array.reset( new db::Array( bla )); } break; \
  case 3: { b_shape3(0) = shape[0]; \
    b_shape3(1) = shape[1]; \
    b_shape3(2) = shape[2]; \
    blitz::Array<T,3> bla = parseArrayData<T,3>( tok, b_shape3 ); \
    array.reset( new db::Array( bla )); } break; \
  case 4: { b_shape4(0) = shape[0]; \
    b_shape4(1) = shape[1]; \
    b_shape4(2) = shape[2]; \
    b_shape4(3) = shape[3]; \
    blitz::Array<T,4> bla = parseArrayData<T,4>( tok, b_shape4 ); \
    array.reset( new db::Array( bla )); } break; \
  default: throw XMLException(); }
   
      switch( elem) {
        case tca::t_bool:
          PARSE_ARRAY_DATA(bool);
          break;
        case tca::t_int8:
          PARSE_ARRAY_DATA(int8_t);
          break;
        case tca::t_int16:
          PARSE_ARRAY_DATA(int16_t);
          break;
        case tca::t_int32:
          PARSE_ARRAY_DATA(int32_t);
          break;
        case tca::t_int64:
          PARSE_ARRAY_DATA(int64_t);
          break;
        case tca::t_uint8:
          PARSE_ARRAY_DATA(uint8_t);
          break;
        case tca::t_uint16:
          PARSE_ARRAY_DATA(uint16_t);
          break;
        case tca::t_uint32:
          PARSE_ARRAY_DATA(uint32_t);
          break;
        case tca::t_uint64:
          PARSE_ARRAY_DATA(uint64_t);
          break;
        case tca::t_float32:
          PARSE_ARRAY_DATA(float);
          break;
        case tca::t_float64:
          PARSE_ARRAY_DATA(double);
          break;
        case tca::t_float128:
          PARSE_ARRAY_DATA(long double);
          break;
        case tca::t_complex64:
          PARSE_ARRAY_DATA(std::complex<float>);
          break;
        case tca::t_complex128:
          PARSE_ARRAY_DATA(std::complex<double>);
          break;
        case tca::t_complex256:
          PARSE_ARRAY_DATA(std::complex<long double>);
          break;
        default:
          throw XMLException();
          break;
      }
    }
    else // External Array
      array.reset( 
        new db::Array(pl.locate(str_filename).string(), str_codecname) );

    // Parse id
    str = xmlGetProp(cur, (const xmlChar*)db::id);
    size_t id = (str!=0? boost::lexical_cast<size_t>((const char*)str): 0);
    TDEBUG3("  Array Id: " << id);
    xmlFree(str);

    return std::make_pair(id, array);
  }


  void XMLParser::parsePathList(const xmlNodePtr cur, db::PathList& pl)
  {
    // Parse the entries
    xmlNodePtr cur_entry = cur->xmlChildrenNode;
    while(cur_entry != 0) { 
      // Parse an entry and add it to the PathList
      if( !strcmp((const char*)cur_entry->name, db::entry) ) {
        // Parse the path of the entry
        xmlChar *str;
        str = xmlGetProp(cur_entry, (const xmlChar*)db::path);
        std::string str_path( str!=0 ? (const char*)str: "" );
        TDEBUG3("  Entry path: " << str_path);
        xmlFree(str);
        // Add it to the path list
        pl.append( str_path );
      }
      cur_entry = cur_entry->next;
    }
  }

}}}

