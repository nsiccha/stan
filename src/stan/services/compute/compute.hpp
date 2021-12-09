#ifndef STAN_SERVICES_COMPUTE_COMPUTE_HPP
#define STAN_SERVICES_COMPUTE_COMPUTE_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
// #include <stan/io/array_var_context.hpp>
#include <stan/services/error_codes.hpp>
// #include <stan/services/util/create_rng.hpp>
// #include <stan/services/util/gq_writer.hpp>
// #include <stan/math/prim/fun/Eigen.hpp>
// #include <boost/algorithm/string.hpp>
// #include <string>
// #include <vector>
#include <stan/model/log_prob_grad.hpp>
#include <fstream>
#include <iostream>

namespace stan {
namespace services {

// template <class Model>
// void get_model_parameters(const Model &model,
//                          std::vector<std::string> &param_names,
//                          std::vector<std::vector<size_t>> &param_dimss) {
//  std::vector<std::string> param_cols;
//  model.constrained_param_names(param_cols, false, false);
//  std::string cur_name("");
//  std::vector<std::string> splits;
//  for (size_t i = 0; i < param_cols.size(); ++i) {
//    boost::algorithm::split(splits, param_cols[i], boost::is_any_of("."));
//    if (splits.size() == 1 || splits[0] != cur_name) {
//      cur_name = splits[0];
//      param_names.emplace_back(cur_name);
//    }
//  }
//  std::vector<std::string> all_param_names;
//  model.get_param_names(all_param_names);
//  size_t num_params = param_names.size();
//  std::vector<std::vector<size_t>> dimss;
//  model.get_dims(dimss);
//  for (size_t i = 0; i < param_names.size(); i++) {
//    for (size_t j = i; j < all_param_names.size(); ++j) {
//      if (param_names[i].compare(all_param_names[j]) == 0) {
//        param_dimss.emplace_back(dimss[j]);
//        break;
//      }
//    }
//  }
// }

  template <typename T>
  void write_line(const T &arg){
    std::cout << arg << std::endl;
  }
  void write_line(const std::vector<double> &arg){
    for(int i = 0; i < arg.size(); ++i){
      std::cout << arg[i];
    }
    std::cout << std::endl;
  }

template <class Model, class Config>
int compute(const Model &model, Config &config,
                        unsigned int seed, callbacks::interrupt &interrupt,
                        callbacks::logger &logger,
                        callbacks::writer &sample_writer) {
  using stan::model::log_prob_grad;

  std::string config_input_path = config.input_path();
  bool config_input_unconstrained = config.input_unconstrained();
  std::string config_output_path = config.output_path();
  bool config_unconstrained_parameters = config.unconstrained_parameters();
  bool config_constrained_parameters = config.constrained_parameters();
  bool config_transformed_parameters = config.transformed_parameters();
  bool config_generated_quantities = config.generated_quantities();
  bool config_constrained_log_probability = config.constrained_log_probability();
  bool config_constrained_log_probability_gradient = config.constrained_log_probability_gradient();
  bool config_unconstrained_log_probability = config.unconstrained_log_probability();
  bool config_unconstrained_log_probability_gradient = config.unconstrained_log_probability_gradient();

  std::vector<std::string> column_names;
  std::vector<std::string> constrained_parameter_names;
  std::vector<std::vector<size_t>> constrained_parameter_dimss;
  std::vector<std::string> unconstrained_parameter_names;
  get_model_parameters(model, constrained_parameter_names, constrained_parameter_dimss);
  // model.constrained_param_names(constrained_parameter_names, false, false);
  model.unconstrained_param_names(unconstrained_parameter_names, false, false);
  size_t no_constrained_parameters = constrained_parameter_names.size();
  size_t no_unconstrained_parameters = unconstrained_parameter_names.size();
  if(config_unconstrained_parameters){
    column_names = unconstrained_parameter_names;
  }
  if(
    config_constrained_parameters
    || config_transformed_parameters
    || config_generated_quantities
  ){
    model.constrained_param_names(
      column_names,
      config_transformed_parameters,
      config_generated_quantities
    );
    if(!config_constrained_parameters){
      column_names.erase(
        column_names.begin(), column_names.begin() + no_constrained_parameters
      );
    }
  }
  // if(config_unconstrained_parameters){
  //   model.unconstrained_param_names(column_names, false, false);
  // }
  // if(config_transformed_parameters){
  // }
  // if(config_generated_quantities){
  // }
  if(config_constrained_log_probability){
    column_names.push_back("lp__");
  }
  if(config_constrained_log_probability_gradient){
    for (int i = 0; i < unconstrained_parameter_names.size(); ++i){
      column_names.push_back(std::string("lpg_") + unconstrained_parameter_names[i]);
    }
  }
  if(config_unconstrained_log_probability){
    column_names.push_back("ulp__");
  }
  if(config_unconstrained_log_probability_gradient){
    for (int i = 0; i < unconstrained_parameter_names.size(); ++i){
      column_names.push_back(std::string("ulpg_") + unconstrained_parameter_names[i]);
    }
  }
  size_t no_output_columns = column_names.size();
  if(no_output_columns == 0){
    logger.warn("No output specified, writing nothing!");
    return error_codes::OK;
  }
  sample_writer(column_names);
  if(config.input_path() == ""){
    logger.info("No input specified, only writing column names!");
    return error_codes::OK;
  }
  std::ifstream ifd(config.input_path(), std::ios::binary);
  ifd.seekg(0, ifd.end);
  size_t size = ifd.tellg();
  ifd.seekg(0);
  std::ofstream ofd(config.output_path(), std::ios::binary);
  size_t no_rows = size / (8 * no_constrained_parameters);
  std::vector<double> constrained_parameters(no_constrained_parameters);
  std::vector<double> unconstrained_parameters(no_unconstrained_parameters);
  std::vector<double> output_row(no_output_columns);

  boost::ecuyer1988 rng = util::create_rng(seed, 1);
  std::vector<int> dummy_params_i;
  std::stringstream msg;
  auto write_vector = [&ofd](const std::vector<double> &values, size_t start){
    ofd.write(
      reinterpret_cast<const char *>(values.data()+start),
      8*(values.size() - start)
    );
  };
  auto write_scalar = [&ofd](double value){
    ofd.write(
      reinterpret_cast<char *>(&value),
      8
    );
  };
  for(size_t row_idx = 0; row_idx < no_rows; ++row_idx){
    if(config.input_unconstrained()){
      ifd.read(
        reinterpret_cast<char*>(unconstrained_parameters.data()),
        8*no_unconstrained_parameters
      );
    }else{
      ifd.read(
        reinterpret_cast<char*>(constrained_parameters.data()),
        8*no_constrained_parameters
      );
      stan::io::array_var_context context(
        constrained_parameter_names, constrained_parameters, constrained_parameter_dimss
      );
      model.transform_inits(context, dummy_params_i, unconstrained_parameters,
                            &msg);
    }
    if(config_unconstrained_parameters){
      write_vector(unconstrained_parameters, 0);
    }
    if(
      config_constrained_parameters
      || config_transformed_parameters
      || config_generated_quantities
    ){
      model.write_array(
        rng,
        unconstrained_parameters,
        dummy_params_i,
        output_row,
        config_transformed_parameters,
        config_generated_quantities,
        &msg
      );
      write_vector(
        output_row,
        config_constrained_parameters ? 0 : no_constrained_parameters
      );
    }
    if(config_constrained_log_probability_gradient){
      double lp = log_prob_grad<false, true>(
        model, unconstrained_parameters, dummy_params_i, output_row, &msg
      );
      if(config_constrained_log_probability){
        write_scalar(lp);
      }
      write_vector(output_row, 0);
    }else if(config_constrained_log_probability){
      double lp = model.template log_prob<false, true>(
        unconstrained_parameters, dummy_params_i, &msg
      );
      write_scalar(lp);
    }
    if(config_unconstrained_log_probability_gradient){
      double lp = log_prob_grad<false, false>(
        model, unconstrained_parameters, dummy_params_i, output_row, &msg
      );
      if(config_unconstrained_log_probability){
        write_scalar(lp);
      }
      write_vector(output_row, 0);
    }else if(config_unconstrained_log_probability){
      double lp = model.template log_prob<false, false>(
        unconstrained_parameters, dummy_params_i, &msg
      );
      write_scalar(lp);
    }
  }
  return error_codes::OK;
}

}  // namespace services
}  // namespace stan
#endif
