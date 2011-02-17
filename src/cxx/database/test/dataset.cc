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
#include "core/cast.h"
#include "database/BinFile.h"
#include "database/Dataset.h"

#include <unistd.h>

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;

  blitz::Array<float,2> d;
  blitz::Array<float,2> e;
  blitz::Array<float,2> f;

  blitz::Array<double,4> g;
  blitz::Array<double,4> h;

  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;

    d.resize(2,2);
    d = 1, 2, 3, 4;
    e.resize(2,2);
    e = 5, 6, 7, 8;

    g.resize(2,3,4,5);
    g = 37.;
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

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
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
void check_equal_4d(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( dbDataset_nodata )
{
  // Create a dataset and save it in the given XML file
  std::string name = "Novel dataset example";
  size_t version = 1;
  Torch::database::Dataset d(name, version);
  // Check that the name and the version are correctly set
  BOOST_CHECK_EQUAL( d.getName().compare(name), 0);
  BOOST_CHECK_EQUAL( d.getVersion(), version);

  // Update the name and the version
  std::string name2 = "Novel dataset example2";
  size_t version2 = 2;  
  d.setName(name2);
  d.setVersion(version2);
  // Check that the name and the version are correctly updated
  BOOST_CHECK_EQUAL( d.getName().compare(name2), 0);
  BOOST_CHECK_EQUAL( d.getVersion(), version2);

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

BOOST_AUTO_TEST_CASE( dbDataset_load_inline_save_inline )
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
  Torch::database::Dataset d(testdata_path.string());
  
  // Save to XML
  std::string tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(d.save(tpx));

  // TODO: check consistency after loading the saved XML database

  //d.consolidateIds();
/*  const std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> > m=d.arraysetIndex();
  for( std::map<size_t, boost::shared_ptr<Torch::database::Arrayset> >::const_iterator 
    it=m.begin(); it!=m.end(); ++it)*/
}

BOOST_AUTO_TEST_CASE( dbDataset_load_inline_save_withexternal )
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
  Torch::database::Dataset d(testdata_path.string());

  // Make the inline arrayset of id 1 an external arrayset
  BOOST_CHECK_NO_THROW(d[1].save( temp_bin_file()));
  // Make the inline array of id 1 of the inline arrayset of id 3 an 
  // external array
  BOOST_CHECK_NO_THROW(d[3][1].save( temp_bin_file()));
  
  // Save to XML
  boost::filesystem::path tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(d.save(tpx.string()));

  // TODO: check consistency after loading the saved XML database
}

BOOST_AUTO_TEST_CASE( dbDataset_load_inline_save_inline_full )
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
  Torch::database::Dataset d(testdata_path.string());
  
  // Save to XML
  std::string tpx = temp_xml_file();
  BOOST_REQUIRE_NO_THROW(d.save(tpx));
}

BOOST_AUTO_TEST_SUITE_END()

