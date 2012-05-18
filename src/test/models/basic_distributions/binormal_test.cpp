#include <stan/mcmc/mcmc_output.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <boost/math/distributions/students_t.hpp>
#include <stdio.h>

class Models_BasicDistributions_Binormal : public ::testing::Test {
protected:
  virtual void SetUp() {
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    path_separator += fgetc(in);
    pclose(in);
    
    model.append("models").append(path_separator);
    model.append("basic_distributions").append(path_separator);
    model.append("binormal");

    output1 = model + "1.csv";
    output2 = model + "2.csv";
    factory.addFile(output1);
    factory.addFile(output2);
    
    expected_y1 = 0.0;
    expected_y2 = 0.0;
  }
  std::string path_separator;
  std::string model;
  std::string output1;
  std::string output2;
  
  stan::mcmc::mcmc_output_factory factory;
  double expected_y1;
  double expected_y2;
};

TEST_F(Models_BasicDistributions_Binormal,RunModel) {
  std::string command;
  command = model;
  command += " --samples=";
  command += output1;
  EXPECT_EQ(0, system(command.c_str())) 
    << "Can not execute command: " << command << std::endl;
            
  
  command = model;
  command += " --samples=";
  command += output2;
  EXPECT_EQ(0, system(command.c_str()))
    << "Can not execute command: " << command << std::endl;
}
/*
TEST_F(Models_BasicDistributions_Binormal, y1) {
  stan::mcmc::mcmc_output y1 = factory.create("y.1");
  double neff = y1.effectiveSize();


  boost::math::students_t t(neff-1.0);
  double T = boost::math::quantile(t, 0.975);
  
  EXPECT_NEAR(expected_y1, y1.mean(), T*sqrt(y1.variance()/neff));
}

TEST_F(Models_BasicDistributions_Binormal, y2) {
  stan::mcmc::mcmc_output y2 = factory.create("y.2");
  double neff = y2.effectiveSize();  

  boost::math::students_t t(neff-1.0);
  double T = boost::math::quantile(t, 0.975);
  
  EXPECT_NEAR(expected_y2, y2.mean(), T*sqrt(y2.variance()/neff));
}
*/
