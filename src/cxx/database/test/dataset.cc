/**
 * @file src/cxx/database/test/arrayset.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Database Arrayset tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DbArrayset Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <vector>

#include <blitz/array.h>
#include <boost/date_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include "core/cast.h"
#include "database/BinFile.h"
#include "database/Dataset.h"
#include "database/Arrayset.h"
#include "database/Array.h"

#include <unistd.h>
#include <iostream>

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;

  blitz::Array<float,2> d;
  blitz::Array<float,2> e;
  blitz::Array<float,2> f;

  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;

    d.resize(2,2);
    d = 1, 2, 3, 4;
    e.resize(2,2);
    e = 5, 6, 7, 8;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary XML filename, and returns the file
 * descriptor
 */
std::string temp_xml_file() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_database_datasetXXXXXX.xml";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

/**
 * @brief Generates a unique temporary .bin filename, and returns the file
 * descriptor
 */
std::string temp_bin_file() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_database_datasetXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

/**
 * @brief Generates a unique temporary .bin filename, and returns the file
 * descriptor
 */
std::string temp_dir() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_database_dataset_dirXXXXXX";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  if( !mkdtemp(char_tpl.get()) )
    throw Torch::core::Exception();
  std::string res = char_tpl.get();
  return res;
}

/**
 * @brief Generates a unique temporary .bin filename in a given directory, and
 * returns the file descriptor
 */
std::string temp_bin_file(const std::string& dir) {
  boost::filesystem::path tpl = dir.c_str();
  tpl /= "torchtest_database_datasetXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

/**
 * @brief Generates a unique temporary XML filename in a given directory, and 
 * returns the file descriptor
 */
std::string temp_xml_file(const std::string& dir) {
  boost::filesystem::path tpl = dir.c_str();
  tpl /= "torchtest_database_datasetXXXXXX.xml";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), Torch::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,3>& a, const blitz::Array<U,3>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        BOOST_CHECK_EQUAL(a(i,j,k), Torch::core::cast<T>(b(i,j,k)));
      }
    }
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  BOOST_REQUIRE_EQUAL(a.extent(3), b.extent(3));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          BOOST_CHECK_EQUAL(a(i,j,k,l), Torch::core::cast<T>(b(i,j,k,l)));
        }
      }
    }
  }
}

void check_equal( const Torch::database::Dataset& ds1, 
  const Torch::database::Dataset& ds2)
{
  // Check the Arraysets
  const std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> > 
    m1 = ds1.arraysetIndex();
  const std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> > 
    m2 = ds2.arraysetIndex();
  BOOST_CHECK_EQUAL( m1.size(), m2.size() );
  for( std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> >::const_iterator 
    it1=m1.begin(), it2=m2.begin(); it1!=m1.end() && it2!=m2.end(); ++it1, ++it2)
  {
    BOOST_CHECK_EQUAL(it1->first, it2->first );
    BOOST_CHECK_EQUAL(it1->second->getRole().compare(it2->second->getRole()), 0 );
    BOOST_CHECK_EQUAL(it1->second->isLoaded(), it2->second->isLoaded() );
    BOOST_CHECK_EQUAL(it1->second->getElementType(), it2->second->getElementType() );
    BOOST_CHECK_EQUAL(it1->second->getNDim(), it2->second->getNDim() );
    const size_t *shape1 = it1->second->getShape();
    const size_t *shape2 = it2->second->getShape();
    for (size_t i=0; i<it1->second->getNDim(); ++i)
      BOOST_CHECK_EQUAL( shape1[i], shape2[i]);
    BOOST_CHECK_EQUAL(it1->second->getNSamples(), it2->second->getNSamples() );
    BOOST_CHECK_EQUAL(it1->second->getFilename().compare(it2->second->getFilename()), 0 );
    BOOST_CHECK_EQUAL(it1->second->getCodec(), it2->second->getCodec() );

    // Check the Arrays
    std::vector<size_t> ids1, ids2;
    it1->second->index(ids1);
    it2->second->index(ids2);
    BOOST_CHECK_EQUAL( ids1.size(), ids2.size() );
    for( std::vector<size_t>::const_iterator
      ita1=ids1.begin(), ita2=ids2.begin(); 
      ita1!=ids1.end() && ita2!=ids2.end(); ++ita1, ++ita2)
    {
      Torch::database::Array a1 = it1->second->operator[](*ita1);
      Torch::database::Array a2 = it2->second->operator[](*ita2);
      BOOST_CHECK_EQUAL(a1.getNDim(), a2.getNDim() );
      BOOST_CHECK_EQUAL(a1.getElementType(), a2.getElementType() );
      const size_t *ashape1 = a1.getShape();
      const size_t *ashape2 = a2.getShape();
      for (size_t i=0; i<a1.getNDim(); ++i)
        BOOST_CHECK_EQUAL( ashape1[i], ashape2[i]);
      BOOST_CHECK_EQUAL(a1.getFilename().compare(a2.getFilename()), 0 );
      BOOST_CHECK_EQUAL(a1.getCodec(), a2.getCodec() );
      BOOST_CHECK_EQUAL(a1.isLoaded(), a2.isLoaded() );

      // Check Array content 
      switch(a1.getNDim())
      {
        case 1: check_equal( a1.cast<std::complex<double>,1>(), 
          a2.cast<std::complex<double>,1>() ); break;
        case 2: check_equal( a1.cast<std::complex<double>,2>(), 
          a2.cast<std::complex<double>,2>() ); break;
        case 3: check_equal( a1.cast<std::complex<double>,3>(), 
          a2.cast<std::complex<double>,3>() ); break;
        case 4: check_equal( a1.cast<std::complex<double>,4>(), 
          a2.cast<std::complex<double>,4>() ); break;
        default: ; break;
      }
    }
  }

  // Check the Relationsets
  const std::map<std::string, boost::shared_ptr<Torch::database::Relationset> > 
    r1 = ds1.relationsetIndex();
  const std::map<std::string, boost::shared_ptr<Torch::database::Relationset> > 
    r2 = ds2.relationsetIndex();
  BOOST_CHECK_EQUAL( r1.size(), r2.size() );
  for( std::map<std::string, boost::shared_ptr<Torch::database::Relationset> >::const_iterator 
    it1=r1.begin(), it2=r2.begin(); it1!=r1.end() && it2!=r2.end(); ++it1, ++it2)
  {
    BOOST_CHECK_EQUAL(it1->first.compare(it2->first), 0 );
    BOOST_CHECK_EQUAL(it1->second->getParent(), &ds1);
    BOOST_CHECK_EQUAL(it2->second->getParent(), &ds2);

    // Check the Rules
    const std::map<std::string, boost::shared_ptr<Torch::database::Rule> > 
      ru1 = it1->second->rules();
    const std::map<std::string, boost::shared_ptr<Torch::database::Rule> > 
      ru2 = it2->second->rules();
    BOOST_CHECK_EQUAL( ru1.size(), ru2.size() );
    for( std::map<std::string, boost::shared_ptr<Torch::database::Rule> >::const_iterator
      itru1=ru1.begin(), itru2=ru2.begin(); 
      itru1!=ru1.end() && itru2!=ru2.end(); ++itru1, ++itru2)
    {
      BOOST_CHECK_EQUAL( itru1->first.compare( itru2->first), 0 );
      BOOST_CHECK_EQUAL( itru1->second->getMin(), itru2->second->getMin() );
      BOOST_CHECK_EQUAL( itru1->second->getMax(), itru2->second->getMax() );
    }

    // Check the Relations
    const std::map<size_t, boost::shared_ptr<Torch::database::Relation> > 
      re1 = it1->second->relations();
    const std::map<size_t, boost::shared_ptr<Torch::database::Relation> > 
      re2 = it2->second->relations();
    BOOST_CHECK_EQUAL( re1.size(), re2.size() );
    for( std::map<size_t, boost::shared_ptr<Torch::database::Relation> >::const_iterator
      itre1=re1.begin(), itre2=re2.begin(); 
      itre1!=re1.end() && itre2!=re2.end(); ++itre1, ++itre2)
    {
      BOOST_CHECK_EQUAL( itre1->first, itre2->first );

      // Check the Members
      const std::list<std::pair<size_t,size_t> > 
        me1 = itre1->second->members();
      const std::list<std::pair<size_t,size_t> > 
        me2 = itre2->second->members();
      BOOST_CHECK_EQUAL( me1.size(), me2.size() );
      for( std::list<std::pair<size_t,size_t> >::const_iterator
        itme1=me1.begin(), itme2=me2.begin();
        itme1!=me1.end() && itme2!=me2.end(); ++itme1, ++itme2)
      {
        BOOST_CHECK_EQUAL( itme1->first, itme2->first );
        BOOST_CHECK_EQUAL( itme1->second, itme2->second );
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( dbDataset_construction )
{
  // Create a dataset and save it in the given XML file
  std::string name = "Novel dataset example";
  size_t version = 1;
  Torch::database::Dataset d(name, version);
  std::string author = "Laurent El Shafey";
  boost::posix_time::ptime dt(boost::gregorian::date(2011,boost::gregorian::Feb,28), 
    boost::posix_time::time_duration(1,2,3));
  d.setAuthor(author);
  d.setDateTime(dt);
  // Check that the name and the version are correctly set
  BOOST_CHECK_EQUAL( d.getName().compare(name), 0);
  BOOST_CHECK_EQUAL( d.getVersion(), version);
  BOOST_CHECK_EQUAL( d.getAuthor(), author);
  bool same_dt = d.getDateTime() == dt;
  BOOST_CHECK_EQUAL( same_dt, true);
//  BOOST_CHECK_EQUAL( d.getDateTime(), dt); // is source of segfault

  // Update the name, the version, the author and the date
  std::string name2 = "Novel dataset example2";
  size_t version2 = 2;  
  std::string author2 = "Laurent T. El Shafey";
  boost::posix_time::ptime dt2(boost::gregorian::date(2011,boost::gregorian::Mar,1),
    boost::posix_time::time_duration(3,2,1));
  d.setName(name2);
  d.setVersion(version2);
  d.setAuthor(author2);
  d.setDateTime(dt2);
  // Check that the name and the version are correctly updated
  BOOST_CHECK_EQUAL( d.getName().compare(name2), 0);
  BOOST_CHECK_EQUAL( d.getVersion(), version2);
  BOOST_CHECK_EQUAL( d.getAuthor(), author2);
  same_dt = d.getDateTime() == dt2;
  BOOST_CHECK_EQUAL( same_dt, true);
//  BOOST_CHECK_EQUAL( d.getDateTime(), dt2); // is source of segfault

  // Check that the next free id is equal to 1
  BOOST_CHECK_EQUAL( d.getNextFreeId(), 1);
  // Consolidate the ids: should not do anything
  d.consolidateIds();
  BOOST_CHECK_EQUAL( d.getNextFreeId(), 1);

  // Get the Arrayset and Relationset map and check that there are empty
  std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> > 
    arraysets = d.arraysetIndex();
  std::map<std::string, boost::shared_ptr<Torch::database::Relationset> > 
    relationsets = d.relationsetIndex();
  BOOST_CHECK_EQUAL(arraysets.empty(),true);
  BOOST_CHECK_EQUAL(relationsets.empty(),true);

  // Check that exists() function is working
  BOOST_CHECK_EQUAL( d.exists(0), false);
  BOOST_CHECK_EQUAL( d.exists("my relationset"), false);

  // Clear the empty Arraysets and Relationsets
  d.clearArraysets();
  d.clearRelationsets();
  BOOST_CHECK_EQUAL(arraysets.empty(),true);
  BOOST_CHECK_EQUAL(relationsets.empty(),true);

  // Call the copy constructor of Dataset and test the member values
  Torch::database::Dataset d2(d);
  // Check that the name and the version are correctly updated
  BOOST_CHECK_EQUAL( d2.getName().compare(d.getName()), 0);
  BOOST_CHECK_EQUAL( d2.getVersion(), d.getVersion());
 
  // Make a copy of the  Dataset and test the member values
  Torch::database::Dataset d3("",1);
  d3 = d;
  // Check that the name and the version are correctly updated
  BOOST_CHECK_EQUAL( d3.getName().compare(d.getName()), 0);
  BOOST_CHECK_EQUAL( d3.getVersion(), d.getVersion());
 
  // Save the Database to XML and load it
  std::string filename = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(d3.save(filename));
  BOOST_REQUIRE_NO_THROW(Torch::database::Dataset d4(filename));
  Torch::database::Dataset d4(filename);

  // Check that the name and the version are correctly set
  BOOST_CHECK_EQUAL( d3.getName().compare(d4.getName()), 0);
  BOOST_CHECK_EQUAL( d3.getVersion(), d4.getVersion());
  // Check that the Arraysets and Relationsets are empty
  BOOST_CHECK_EQUAL(d4.arraysetIndex().empty(),true);
  BOOST_CHECK_EQUAL(d4.relationsetIndex().empty(),true);
}


BOOST_AUTO_TEST_CASE( dbDataset_arrayset )
{
  // Create a dataset and save it in the given XML file
  std::string name = "Novel dataset example";
  size_t version = 1;
  Torch::database::Dataset ds(name, version);

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a1(
    new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_c1(
    new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec1;
  vec1.push_back(db_a1);
  vec1.push_back(db_c1);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar1(vec1);

  // Add the arrayset to the dataset and check the id
  size_t id;
  BOOST_REQUIRE_NO_THROW( id = ds.add(db_Ar1) );
  BOOST_CHECK_EQUAL( id, 1 );

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a2(
    new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_c2(
    new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec2;
  vec2.push_back(db_c2);
  vec2.push_back(db_a2);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar2(vec2);

  // Add the arrayset to the dataset
  BOOST_REQUIRE_NO_THROW( ds.add(3, db_Ar2) );

  // Access Arrayset 1 and 3 and check content
  check_equal( ds[1][1].get<double,1>(), a);
  check_equal( ds[1][2].get<double,1>(), c);
  check_equal( ds[3][1].get<double,1>(), c);
  check_equal( ds[3][2].get<double,1>(), a);
  check_equal( ds.ptr(3)->operator[](2).get<double,1>(), c);

  // Add an Arrayset at an occupied position and check that an exception
  // is thrown.
  BOOST_CHECK_THROW( ds.add(3, db_Ar2), Torch::database::IndexError );

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_d3(
    new Torch::database::Array(d));
  boost::shared_ptr<Torch::database::Array> db_e3(
    new Torch::database::Array(e));
  boost::shared_ptr<Torch::database::Array> db_f3(
    new Torch::database::Array(e));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec3;
  vec3.push_back(db_d3);
  vec3.push_back(db_e3);
  vec3.push_back(db_f3);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar3(vec3);

  // Set the arrayset at a non occupied position and check that an exception
  // is thrown.
  BOOST_CHECK_THROW( ds.set(2, db_Ar3), Torch::database::IndexError );
  // Set the arrayset at an occupied position and check that the dataset is
  // updated.
  BOOST_CHECK_NO_THROW( ds.set(1, db_Ar3) );
  check_equal( ds[1][1].get<float,2>(), d);
  check_equal( ds[1][2].get<float,2>(), e);
  check_equal( ds[1][3].get<float,2>(), e);
  check_equal( ds.ptr(1)->operator[](3).get<float,2>(), e);

  // Check that the Arrayset of id 3 exists and that the next free id is 4
  BOOST_CHECK_EQUAL( ds.exists(3), true);
  BOOST_CHECK_EQUAL( ds.exists(2), false);
  BOOST_CHECK_EQUAL( ds.getNextFreeId(), 4);
  BOOST_CHECK_EQUAL( ds.arraysetIndex().size(), 2 );

  // Consolidate the id: 3 is moved to 2
  BOOST_CHECK_NO_THROW( ds.consolidateIds() );

  // Remove Arrayset of id 2
  BOOST_CHECK_NO_THROW( ds.remove(2) );
  BOOST_CHECK_EQUAL( ds.exists(2), false);
  BOOST_CHECK_EQUAL( ds.getNextFreeId(), 2);
  BOOST_CHECK_EQUAL( ds.arraysetIndex().size(), 1 );

  // Clear Arraysets
  BOOST_CHECK_NO_THROW( ds.clearArraysets() );
  BOOST_CHECK_EQUAL( ds.getNextFreeId(), 1);
  BOOST_CHECK_EQUAL( ds.arraysetIndex().size(), 0 );
}


BOOST_AUTO_TEST_CASE( dbDataset_relationset )
{
  // Create a dataset and save it in the given XML file
  std::string name = "Novel dataset example";
  size_t version = 1;
  Torch::database::Dataset ds(name, version);

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a1(
    new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_c1(
    new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec1;
  vec1.push_back(db_a1);
  vec1.push_back(db_c1);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar1(vec1);

  // Create the relationsets
  Torch::database::Relationset db_R1;
  std::string role1("rule1");
  BOOST_CHECK_NO_THROW( db_R1.add(role1, Torch::database::Rule(0,1)));
  Torch::database::Relationset db_R2;
  std::string role2("rule2");
  BOOST_CHECK_NO_THROW( db_R2.add(role2, Torch::database::Rule(0,1)));
  Torch::database::Relationset db_R3;
  std::string role3("rule3");
  BOOST_CHECK_NO_THROW( db_R3.add(role3, Torch::database::Rule(0,1)));
  std::string r_name1("MyRelationset1");
  std::string r_name2("MyRelationset2");
  std::string r_name3("MyRelationset3");

  // Add/Set/Remove them to/from the dataset and perform the check
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 0);
  BOOST_CHECK_EQUAL( ds.exists(r_name1), false);
  BOOST_CHECK_NO_THROW( ds.add(r_name1,db_R1) );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 1);
  BOOST_CHECK_EQUAL( ds.exists(r_name1), true);
  BOOST_CHECK_THROW( ds.add(r_name1,db_R2), Torch::database::NameError );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 1);
  BOOST_CHECK_EQUAL( ds.exists(r_name1), true);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), false);
  BOOST_CHECK_NO_THROW( ds.add(r_name2,db_R2) );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 2);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), true);
  BOOST_CHECK_EQUAL( ds.exists(r_name3), false);
  BOOST_CHECK_THROW( ds.set(r_name3,db_R3), Torch::database::NameError );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 2);
  BOOST_CHECK_EQUAL( ds.exists(r_name3), false);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), true);
  BOOST_CHECK_NO_THROW( ds.set(r_name2,db_R3) );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 2);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), true);
  BOOST_CHECK_NO_THROW( ds.remove(r_name2) );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 1);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), false);

  // Clear the relationsets
  BOOST_CHECK_NO_THROW( ds.clearRelationsets() );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 0);
  BOOST_CHECK_EQUAL( ds.exists(r_name1), false);
  BOOST_CHECK_EQUAL( ds.exists(r_name2), false);
  BOOST_CHECK_EQUAL( ds.exists(r_name3), false);

  // Access the relationset
  BOOST_CHECK_NO_THROW( ds.add(r_name1,db_R1) );
  BOOST_CHECK_EQUAL( ds.relationsetIndex().size(), 1);
  BOOST_CHECK_NO_THROW( ds[r_name1][role1] );
  BOOST_CHECK_NO_THROW( ds.ptr(r_name1)->operator[](role1) );
}


BOOST_AUTO_TEST_CASE( dbDataset_parsewrite_inline )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "db_inline.xml";

  // Load from XML
  BOOST_REQUIRE_NO_THROW(Torch::database::Dataset ds1(testdata_path.string()));
  Torch::database::Dataset ds1(testdata_path.string());
  
  // Save to XML
  std::string tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(ds1.save(tpx));

  // Load and parse the saved XML database
  Torch::database::Dataset ds2(tpx);

  // Check that the Datasets are similar
  check_equal( ds1, ds2);
}


BOOST_AUTO_TEST_CASE( dbDataset_parsewrite_inline2 )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "db_inline2.xml";

  // Load from XML
  BOOST_REQUIRE_NO_THROW(Torch::database::Dataset d(testdata_path.string()));
  Torch::database::Dataset ds1(testdata_path.string());
  
  // Save to XML
  std::string tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(ds1.save(tpx));

  // Load and parse the saved XML database
  Torch::database::Dataset ds2(tpx);
  
  // Check that the Datasets are similar
  check_equal( ds1, ds2);
}


BOOST_AUTO_TEST_CASE( dbDataset_parsewrite_withexternal )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "db_inline.xml";

  // Load from XML
  BOOST_REQUIRE_NO_THROW(Torch::database::Dataset d(testdata_path.string()));
  Torch::database::Dataset ds1(testdata_path.string());

  // Make the inline arrayset of id 1 an external arrayset
  BOOST_CHECK_NO_THROW(ds1[1].save( temp_bin_file()));
  // Make the inline array of id 1 of the inline arrayset of id 3 an 
  // external array
  BOOST_CHECK_NO_THROW(ds1[3][1].save( temp_bin_file()));
  
  // Save to XML
  std::string tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(ds1.save(tpx));

  // Load and parse the saved XML database
  Torch::database::Dataset ds2(tpx);
  
  // Check that the Datasets are similar
  check_equal( ds1, ds2);
}


BOOST_AUTO_TEST_CASE( dbDataset_pathlist2 )
{
  // Make the inline array of id 1 of the inline arrayset of id 3 an 
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "db_externalarrayset.xml";
  std::cout << testdata_path << std::endl;

  // Load from XML
  BOOST_REQUIRE_NO_THROW(Torch::database::Dataset d(testdata_path.string()));
  Torch::database::Dataset ds1(testdata_path.string());

  // Access Arrayset 1 and 3 and check content
  check_equal( ds1[1][1].get<double,1>(), a);
  check_equal( ds1[1][2].get<double,1>(), c);
  check_equal( ds1[2][1].get<double,1>(), c);
  check_equal( ds1[2][2].get<double,1>(), a);  

  // Define a temporary directory to store external data and add it to the
  // PathList of the dataset
  std::string path_dir = temp_dir();
  std::string dataset_xml2( temp_xml_file( path_dir) );
  ds1.save( dataset_xml2 );

/*  // Add a relative path in the pathlist 
  Torch::database::Dataset ds2 = ds1;
  Torch::database::PathList pl2 = ds2.getPathList();
  pl2.setCurrentDirectory( path_dir );


  boost::filesystem::path data_dir( path_dir );
  data_dir /= "data";
  boost::filesystem::create_directory( data_dir );
  std::cout << data_dir << std::endl;
  Torch::database::PathList pl( "data/" );
  ds2.setPathList( pl );*/
}


BOOST_AUTO_TEST_SUITE_END()

