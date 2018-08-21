// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_HDF5

#  include <deal.II/base/hdf5.h>

#  include <deal.II/lac/full_matrix.h>

#  include <hdf5.h>

#  include <memory>
#  include <numeric>
#  include <vector>

DEAL_II_NAMESPACE_OPEN

namespace HDF5

{
  namespace internal
  {
    // This function gives the HDF5 datatype corresponding to the C++ type. In
    // the case of std::complex types the HDF5 handlers are automatically freed
    // using the destructor of std::shared_ptr.
    template <typename number>
    std::shared_ptr<hid_t>
    get_hdf5_datatype()
    {
      std::shared_ptr<hid_t> t_type;
      if (std::is_same<number, float>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t);
          *t_type = H5T_NATIVE_FLOAT;
        }
      else if (std::is_same<number, double>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t);
          *t_type = H5T_NATIVE_DOUBLE;
        }
      else if (std::is_same<number, int>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t);
          *t_type = H5T_NATIVE_INT;
        }
      else if (std::is_same<number, unsigned int>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t);
          *t_type = H5T_NATIVE_UINT;
        }
      else if (std::is_same<number, std::complex<float>>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
            // Relase the HDF5 resource
            H5Tclose(*pointer);
            delete pointer;
          });
          *t_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
          //  The C++ standards committee agreed to mandate that the storage
          //  format used for the std::complex type be binary-compatible with
          //  the C99 type, i.e. an array T[2] with consecutive real [0] and
          //  imaginary [1] parts.
          H5Tinsert(*t_type, "r", 0, H5T_NATIVE_FLOAT);
          H5Tinsert(*t_type, "i", sizeof(float), H5T_NATIVE_FLOAT);
        }
      else if (std::is_same<number, std::complex<double>>::value)
        {
          t_type  = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
            // Relase the HDF5 resource
            H5Tclose(*pointer);
            delete pointer;
          });
          *t_type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
          //  The C++ standards committee agreed to mandate that the storage
          //  format used for the std::complex type be binary-compatible with
          //  the C99 type, i.e. an array T[2] with consecutive real [0] and
          //  imaginary [1] parts.
          H5Tinsert(*t_type, "r", 0, H5T_NATIVE_DOUBLE);
          H5Tinsert(*t_type, "i", sizeof(double), H5T_NATIVE_DOUBLE);
        }
      else
        {
          Assert(false, ExcInternalError());
        }
      return t_type;
    }

    // This function returns the pointer to the raw data of a container
    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, std::vector<number>>::value,
      unsigned int>::type
    get_container_size(const Container<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong
      return static_cast<unsigned int>(data.size());
    }

    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, FullMatrix<number>>::value,
      unsigned int>::type
    get_container_size(const Container<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong.
      // Use the first element of FullMatrix to get the pointer to the raw data
      return static_cast<unsigned int>(data.m() * data.n());
    }

    // This function returns the pointer to the raw data of a container
    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, std::vector<number>>::value,
      void *>::type
    get_container_pointer(Container<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong
      return data.data();
    }

    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, FullMatrix<number>>::value,
      void *>::type
    get_container_pointer(FullMatrix<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong.
      // Use the first element of FullMatrix to get the pointer to the raw data
      return &data[0][0];
    }

    // This function returns the pointer to the raw data of a container
    // The returned pointer is const, which means that it can be used only to
    // read the data
    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, std::vector<number>>::value,
      const void *>::type
    get_container_const_pointer(const Container<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong
      return data.data();
    }

    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, FullMatrix<number>>::value,
      const void *>::type
    get_container_const_pointer(const FullMatrix<number> &data)
    {
      // It is very important to pass the variable "data" by reference otherwise
      // the pointer will be wrong.
      // Use the first element of FullMatrix to get the pointer to the raw data
      return &data[0][0];
    }

    // This function initializes a container of number type
    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, std::vector<number>>::value,
      Container<number>>::type
    initialize_container(std::vector<hsize_t> dimensions)
    {
      return std::vector<number>(std::accumulate(
        dimensions.begin(), dimensions.end(), 1, std::multiplies<int>()));
    }

    template <template <class...> class Container, typename number>
    typename std::enable_if<
      std::is_same<Container<number>, FullMatrix<number>>::value,
      Container<number>>::type
    initialize_container(std::vector<hsize_t> dimensions)
    {
      return FullMatrix<number>(dimensions[0], dimensions[1]);
    }

    // Convert a HDF5 no_collective_cause code to a human readable string
    std::string
    no_collective_cause_to_string(uint32_t no_collective_cause)
    {
      std::string message;
      // The first is not a bitmask compararison, the rest are bitmask
      // comparisons.
      // https://support.hdfgroup.org/HDF5/doc/RM/RM_H5P.html#Property-GetMpioNoCollectiveCause
      // See H5Ppublic.h
      // Hex codes are used because the HDF5 Group can deprecate some of the
      // enum codes. For example the enum code H5D_MPIO_FILTERS is not defined
      // in 1.10.2 because it is possible to use compressed datasets with the
      // MPI/IO driver.

      // H5D_MPIO_COLLECTIVE
      if (no_collective_cause == 0x00)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_COLLECTIVE";
        }
      // H5D_MPIO_SET_INDEPENDENT
      if ((no_collective_cause & 0x01) == 0x01)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_SET_INDEPENDENT";
        }
      // H5D_MPIO_DATATYPE_CONVERSION
      if ((no_collective_cause & 0x02) == 0x02)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_DATATYPE_CONVERSION";
        }
      // H5D_MPIO_DATA_TRANSFORMS
      if ((no_collective_cause & 0x04) == 0x04)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_DATA_TRANSFORMS";
        }
      // H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES
      if ((no_collective_cause & 0x10) == 0x10)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES";
        }
      // H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET
      if ((no_collective_cause & 0x20) == 0x20)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET";
        }
      // H5D_MPIO_FILTERS
      if ((no_collective_cause & 0x40) == 0x40)
        {
          if (message.length() > 0)
            {
              message += ", ";
            }
          message += "H5D_MPIO_FILTERS";
        }
      return message;
    }
  } // namespace internal


  HDF5Object::HDF5Object(const std::string name, bool mpi)
    : name(name)
    , mpi(mpi)
  {}

  template <typename T>
  T
  HDF5Object::attr(const std::string attr_name) const
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<T>();
    T                      value;
    hid_t                  attr;


    attr = H5Aopen(*hdf5_reference, attr_name.data(), H5P_DEFAULT);
    H5Aread(attr, *t_type, &value);
    H5Aclose(attr);
    return value;
  }

  template <>
  bool
  HDF5Object::attr(const std::string attr_name) const
  {
    // The enum field generated by h5py can be casted to int
    int   int_value;
    hid_t attr;
    attr = H5Aopen(*hdf5_reference, attr_name.data(), H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &int_value);
    H5Aclose(attr);
    // The int can be casted to a bool
    bool bool_value = (bool)int_value;
    return bool_value;
  }

  template <>
  std::string
  HDF5Object::attr(const std::string attr_name) const
  {
    // Reads a UTF8 variable string
    //
    // code inspired from
    // https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/vlstratt.c
    //
    // In the case of a variable length string the user does not have to reserve
    // memory for string_out. The call HAread will reserve the memory and the
    // user has to free the memory.
    //
    // Todo:
    // - Use H5Dvlen_reclaim instead of free

    char * string_out;
    hid_t  attr;
    hid_t  type;
    herr_t ret;

    /* Create a datatype to refer to. */
    type = H5Tcopy(H5T_C_S1);
    Assert(type >= 0, ExcInternalError());

    // Python strings are encoded in UTF8
    ret = H5Tset_cset(type, H5T_CSET_UTF8);
    Assert(type >= 0, ExcInternalError());

    ret = H5Tset_size(type, H5T_VARIABLE);
    Assert(ret >= 0, ExcInternalError());

    attr = H5Aopen(*hdf5_reference, attr_name.data(), H5P_DEFAULT);
    Assert(attr >= 0, ExcInternalError());

    ret = H5Aread(attr, type, &string_out);
    Assert(ret >= 0, ExcInternalError());

    std::string string_value(string_out);
    // The memory of the variable length string has to be freed.
    // H5Dvlen_reclaim could be also used
    free(string_out);
    H5Tclose(type);
    H5Aclose(attr);
    return string_value;
  }

  template <typename T>
  void
  HDF5Object::write_attr(const std::string attr_name, const T value) const
  {
    hid_t                  attr;
    hid_t                  aid;
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<T>();


    /*
     * Create scalar attribute.
     */
    aid  = H5Screate(H5S_SCALAR);
    attr = H5Acreate2(*hdf5_reference,
                      attr_name.data(),
                      *t_type,
                      aid,
                      H5P_DEFAULT,
                      H5P_DEFAULT);

    /*
     * Write scalar attribute.
     */
    H5Awrite(attr, *t_type, &value);

    H5Sclose(aid);
    H5Aclose(attr);
  }

  template <>
  void
  HDF5Object::write_attr(const std::string attr_name,
                         const std::string value) const
  {
    // Writes a UTF8 variable string
    //
    // code inspired from
    // https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/vlstratt.c
    //
    // In the case of a variable length string, H5Awrite needs the address of a
    // (char *). For this reason the std::string value has been copied to a C
    // string.

    hid_t  attr;
    hid_t  aid;
    hid_t  t_type;
    herr_t ret;

    // Reserve space for the string and the null terminator
    char *c_string_value = (char *)malloc(sizeof(char) * (value.size() + 1));
    strcpy(c_string_value, value.data());

    /* Create a datatype to refer to. */
    t_type = H5Tcopy(H5T_C_S1);
    Assert(t_type >= 0, ExcInternalError());

    // Python strings are encoded in UTF8
    ret = H5Tset_cset(t_type, H5T_CSET_UTF8);
    Assert(t_type >= 0, ExcInternalError());

    ret = H5Tset_size(t_type, H5T_VARIABLE);
    Assert(ret >= 0, ExcInternalError());

    /*
     * Create scalar attribute.
     */
    aid  = H5Screate(H5S_SCALAR);
    attr = H5Acreate2(
      *hdf5_reference, attr_name.data(), t_type, aid, H5P_DEFAULT, H5P_DEFAULT);

    /*
     * Write scalar attribute.
     */
    ret = H5Awrite(attr, t_type, &c_string_value);
    Assert(ret >= 0, ExcInternalError());

    free(c_string_value);
    H5Sclose(aid);
    H5Aclose(attr);
  }

  DataSet::DataSet(const std::string name,
                   const hid_t &     parent_group_id,
                   const bool        mpi)
    : HDF5Object(name, mpi)
    , _check_io_mode(false)
    , _io_mode(H5D_MPIO_NO_COLLECTIVE)
    , _local_no_collective_cause(H5D_MPIO_SET_INDEPENDENT)
    , _global_no_collective_cause(H5D_MPIO_SET_INDEPENDENT)
  {
    hdf5_reference = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Dclose(*pointer);
      delete pointer;
    });
    dataspace      = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Sclose(*pointer);
      delete pointer;
    });

    *hdf5_reference = H5Dopen2(parent_group_id, name.data(), H5P_DEFAULT);
    *dataspace      = H5Dget_space(*hdf5_reference);
    auto rank_ret   = H5Sget_simple_extent_ndims(*dataspace);
    // rank_ret can take a negative value if the function fails. rank is
    // unsigned int, that is way rank_ret is used to store the return
    // value of H5Sget_simple_extent_ndims
    Assert(rank_ret >= 0, ExcInternalError());
    _rank         = rank_ret;
    hsize_t *dims = (hsize_t *)malloc(_rank * sizeof(hsize_t));
    rank_ret      = H5Sget_simple_extent_dims(*dataspace, dims, NULL);
    Assert(rank_ret == static_cast<int>(_rank), ExcInternalError());
    _dimensions.assign(dims, dims + _rank);
    free(dims);

    _size = 1;
    for (auto &&dimension : _dimensions)
      {
        _size *= dimension;
      }
  }

  DataSet::DataSet(const std::string      name,
                   const hid_t &          parent_group_id,
                   std::vector<hsize_t>   dimensions,
                   std::shared_ptr<hid_t> t_type,
                   const bool             mpi)
    : HDF5Object(name, mpi)
    , _rank(dimensions.size())
    , _dimensions(dimensions)
    , _check_io_mode(false)
    , _io_mode(H5D_MPIO_NO_COLLECTIVE)
    , _local_no_collective_cause(H5D_MPIO_SET_INDEPENDENT)
    , _global_no_collective_cause(H5D_MPIO_SET_INDEPENDENT)
  {
    hdf5_reference = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Dclose(*pointer);
      delete pointer;
    });
    dataspace      = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Sclose(*pointer);
      delete pointer;
    });

    *dataspace = H5Screate_simple(_rank, dimensions.data(), NULL);

    *hdf5_reference = H5Dcreate2(parent_group_id,
                                 name.data(),
                                 *t_type,
                                 *dataspace,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    _size = 1;
    for (auto &&dimension : dimensions)
      {
        _size *= dimension;
      }
  }

  template <template <class...> class Container, typename number>
  Container<number>
  DataSet::read()
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    hid_t                  plist;
    herr_t                 ret;

    Container<number> data =
      internal::initialize_container<Container, number>(_dimensions);

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcInternalError());
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcInternalError());
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    ret = H5Dread(*hdf5_reference,
                  *t_type,
                  H5S_ALL,
                  H5S_ALL,
                  plist,
                  internal::get_container_pointer<Container, number>(data));
    Assert(ret >= 0, ExcInternalError());

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcInternalError());
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0, ExcInternalError());
          }
        ret = H5Pclose(plist);
        Assert(ret >= 0, ExcInternalError());
      }

    return data;
  }

  template <typename number>
  std::vector<number>
  DataSet::read_selection(const std::vector<hsize_t> coordinates)
  {
    Assert(coordinates.size() % _rank == 0,
           ExcMessage(
             "The dimension of coordinates has to be divisible by the rank"));
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    hid_t                  plist;
    hid_t                  memory_dataspace;
    herr_t                 ret;

    std::vector<hsize_t> data_dimensions{
      static_cast<hsize_t>(coordinates.size() / _rank)};

    std::vector<number> data(
      static_cast<unsigned int>(coordinates.size() / _rank));

    memory_dataspace = H5Screate_simple(1, data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_elements(*dataspace,
                             H5S_SELECT_SET,
                             data.size(),
                             coordinates.data());
    Assert(ret >= 0, ExcMessage("Error at H5Sselect_elements"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    ret = H5Dread(*hdf5_reference,
                  *t_type,
                  memory_dataspace,
                  *dataspace,
                  plist,
                  data.data());
    Assert(ret >= 0, ExcMessage("Error at H5Dread"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        ret = H5Pclose(plist);
        Assert(ret >= 0, ExcMessage("Error at H5Pclose"));
      }

    ret = H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5SClose"));

    return data;
  }

  template <template <class...> class Container, typename number>
  Container<number>
  DataSet::read_hyperslab(const std::vector<hsize_t> &offset,
                          const std::vector<hsize_t> &count)
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    hid_t                  plist;
    hid_t                  memory_dataspace;
    herr_t                 ret;

    // In this particular overload of read_hyperslab the data_dimensions are
    // the same as count
    std::vector<hsize_t> data_dimensions = count;

    Container<number> data =
      internal::initialize_container<Container, number>(data_dimensions);

    memory_dataspace =
      H5Screate_simple(data_dimensions.size(), data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_hyperslab(
      *dataspace, H5S_SELECT_SET, offset.data(), NULL, count.data(), NULL);
    Assert(ret >= 0, ExcMessage("Error at H5Sselect_hyperslab"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    ret = H5Dread(*hdf5_reference,
                  *t_type,
                  memory_dataspace,
                  *dataspace,
                  plist,
                  internal::get_container_pointer<Container, number>(data));
    Assert(ret >= 0, ExcMessage("Error at H5Dread"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        ret = H5Pclose(plist);
        Assert(ret >= 0, ExcMessage("Error at H5Pclose"));
      }

    ret = H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5SClose"));

    return data;
  }

  template <typename number>
  void
  DataSet::read_none()
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    std::vector<hsize_t>   data_dimensions = {0};

    hid_t  memory_dataspace;
    hid_t  plist;
    herr_t ret;

    memory_dataspace = H5Screate_simple(1, data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_none(*dataspace);
    Assert(ret >= 0, ExcMessage("H5Sselect_none"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    // The pointer of data can safely be NULL, see the discussion at the HDF5
    // forum:
    // https://forum.hdfgroup.org/t/parallel-i-o-does-not-support-filters-yet/884/17
    ret = H5Dread(
      *hdf5_reference, *t_type, memory_dataspace, *dataspace, plist, NULL);
    Assert(ret >= 0, ExcMessage("Error at H5Dread"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        ret = H5Pclose(plist);
        Assert(ret >= 0, ExcMessage("Error at H5Pclose"));
      }

    ret = H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5SClose"));
  }

  template <template <class...> class Container, typename number>
  void
  DataSet::write(const Container<number> &data)
  {
    AssertDimension(_size, internal::get_container_size(data));
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    hid_t                  plist;
    herr_t                 ret;

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    H5Dwrite(*hdf5_reference,
             *t_type,
             H5S_ALL,
             H5S_ALL,
             plist,
             internal::get_container_const_pointer<Container, number>(data));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcInternalError());
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0, ExcInternalError());
          }
        H5Pclose(plist);
      }
  }

  template <typename number>
  void
  DataSet::write_selection(const std::vector<number> &data,
                           const std::vector<hsize_t> coordinates)
  {
    AssertDimension(coordinates.size(), data.size() * _rank);
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    std::vector<hsize_t>   data_dimensions = {data.size()};

    hid_t  memory_dataspace;
    hid_t  plist;
    herr_t ret;


    memory_dataspace = H5Screate_simple(1, data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_elements(*dataspace,
                             H5S_SELECT_SET,
                             data.size(),
                             coordinates.data());
    Assert(ret >= 0, ExcMessage("Error at H5Sselect_elements"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    ret = H5Dwrite(*hdf5_reference,
                   *t_type,
                   memory_dataspace,
                   *dataspace,
                   plist,
                   data.data());
    Assert(ret >= 0, ExcMessage("Error at H5Dwrite"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        H5Pclose(plist);
      }
    ret = H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5SClose"));
  }

  template <template <class...> class Container, typename number>
  void
  DataSet::write_hyperslab(const Container<number> &   data,
                           const std::vector<hsize_t> &offset,
                           const std::vector<hsize_t> &count)
  {
    AssertDimension(std::accumulate(count.begin(),
                                    count.end(),
                                    1,
                                    std::multiplies<unsigned int>()),
                    internal::get_container_size(data));
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    // In this particular overload of write_hyperslab the data_dimensions are
    // the same as count
    std::vector<hsize_t> data_dimensions = count;

    hid_t  memory_dataspace;
    hid_t  plist;
    herr_t ret;

    memory_dataspace =
      H5Screate_simple(data_dimensions.size(), data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_hyperslab(
      *dataspace, H5S_SELECT_SET, offset.data(), NULL, count.data(), NULL);
    Assert(ret >= 0, ExcMessage("Error at H5Sselect_hyperslab"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }
    ret =
      H5Dwrite(*hdf5_reference,
               *t_type,
               memory_dataspace,
               *dataspace,
               plist,
               internal::get_container_const_pointer<Container, number>(data));
    Assert(ret >= 0, ExcMessage("Error at H5Dwrite"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        H5Pclose(plist);
        Assert(ret >= 0, ExcMessage("Error at H5Pclose"));
      }
    H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5Sclose"));
  }

  template <template <class...> class Container, typename number>
  void
  DataSet::write_hyperslab(const Container<number> &   data,
                           const std::vector<hsize_t> &data_dimensions,
                           const std::vector<hsize_t> &offset,
                           const std::vector<hsize_t> &stride,
                           const std::vector<hsize_t> &count,
                           const std::vector<hsize_t> &block)
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();

    hid_t  memory_dataspace;
    hid_t  plist;
    herr_t ret;

    memory_dataspace =
      H5Screate_simple(data_dimensions.size(), data_dimensions.data(), NULL);
    Assert(memory_dataspace >= 0, ExcMessage("Error at H5Screate_simple"));
    ret = H5Sselect_hyperslab(*dataspace,
                              H5S_SELECT_SET,
                              offset.data(),
                              stride.data(),
                              count.data(),
                              block.data());
    Assert(ret >= 0, ExcMessage("Error at H5Sselect_hyperslab"));

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        Assert(plist >= 0, ExcMessage("Error at H5Pcreate"));
        ret = H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
        Assert(ret >= 0, ExcMessage("Error at H5Pset_dxpl_mpio"));
      }
    else
      {
        plist = H5P_DEFAULT;
      }
    ret =
      H5Dwrite(*hdf5_reference,
               *t_type,
               memory_dataspace,
               *dataspace,
               plist,
               internal::get_container_const_pointer<Container, number>(data));
    Assert(ret >= 0, ExcMessage("Error at H5Dwrite"));

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcMessage("Error at H5Pget_mpio_actual_io_mode"));
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0,
                   ExcMessage("Error at H5Pget_mpio_no_collective_cause"));
          }
        H5Pclose(plist);
        Assert(ret >= 0, ExcMessage("Error at H5Pclose"));
      }
    H5Sclose(memory_dataspace);
    Assert(ret >= 0, ExcMessage("Error at H5Sclose"));
  }

  template <typename number>
  void
  DataSet::write_none()
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    std::vector<hsize_t>   data_dimensions = {0};

    hid_t  memory_dataspace;
    hid_t  plist;
    herr_t ret;

    memory_dataspace = H5Screate_simple(1, data_dimensions.data(), NULL);
    H5Sselect_none(*dataspace);

    if (mpi)
      {
        plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    // The pointer of data can safely be NULL, see the discussion at the HDF5
    // forum:
    // https://forum.hdfgroup.org/t/parallel-i-o-does-not-support-filters-yet/884/17
    H5Dwrite(
      *hdf5_reference, *t_type, memory_dataspace, *dataspace, plist, NULL);

    if (mpi)
      {
        if (_check_io_mode)
          {
            ret = H5Pget_mpio_actual_io_mode(plist, &_io_mode);
            Assert(ret >= 0, ExcInternalError());
            ret = H5Pget_mpio_no_collective_cause(plist,
                                                  &_local_no_collective_cause,
                                                  &_global_no_collective_cause);
            Assert(ret >= 0, ExcInternalError());
          }
        H5Pclose(plist);
      }
    H5Sclose(memory_dataspace);
  }

  template <>
  H5D_mpio_actual_io_mode_t
  DataSet::io_mode()
  {
    Assert(_check_io_mode,
           ExcMessage(
             "check_io_mode() should be true in order to use io_mode()"));
    return _io_mode;
  }



  template <>
  std::string
  DataSet::io_mode()
  {
    Assert(_check_io_mode,
           ExcMessage(
             "check_io_mode() should be true in order to use io_mode()"));
    switch (_io_mode)
      {
        case (H5D_MPIO_NO_COLLECTIVE):
          return std::string("H5D_MPIO_NO_COLLECTIVE");
          break;
        case (H5D_MPIO_CHUNK_INDEPENDENT):
          return std::string("H5D_MPIO_CHUNK_INDEPENDENT");
          break;
        case (H5D_MPIO_CHUNK_COLLECTIVE):
          return std::string("H5D_MPIO_CHUNK_COLLECTIVE");
          break;
        case (H5D_MPIO_CHUNK_MIXED):
          return std::string("H5D_MPIO_CHUNK_MIXED");
          break;
        case (H5D_MPIO_CONTIGUOUS_COLLECTIVE):
          return std::string("H5D_MPIO_CONTIGUOUS_COLLECTIVE");
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }

  template <>
  uint32_t
  DataSet::local_no_collective_cause()
  {
    Assert(
      _check_io_mode,
      ExcMessage(
        "check_io_mode() should be true in order to use local_no_collective_cause()"));
    return _local_no_collective_cause;
  }



  template <>
  std::string
  DataSet::local_no_collective_cause()
  {
    Assert(
      _check_io_mode,
      ExcMessage(
        "check_io_mode() should be true in order to use local_no_collective_cause()"));
    return internal::no_collective_cause_to_string(_local_no_collective_cause);
  }

  template <>
  uint32_t
  DataSet::global_no_collective_cause()
  {
    Assert(
      _check_io_mode,
      ExcMessage(
        "check_io_mode() should be true in order to use global_no_collective_cause()"));
    return _global_no_collective_cause;
  }



  template <>
  std::string
  DataSet::global_no_collective_cause()
  {
    Assert(
      _check_io_mode,
      ExcMessage(
        "check_io_mode() should be true in order to use global_no_collective_cause()"));
    return internal::no_collective_cause_to_string(_global_no_collective_cause);
  }

  bool
  DataSet::check_io_mode() const
  {
    return _check_io_mode;
  }

  void
  DataSet::check_io_mode(bool check_io_mode)
  {
    _check_io_mode = check_io_mode;
  }

  std::vector<hsize_t>
  DataSet::dimensions() const
  {
    return _dimensions;
  }

  unsigned int
  DataSet::size() const
  {
    return _size;
  }

  unsigned int
  DataSet::rank() const
  {
    return _rank;
  }

  Group::Group(const std::string name,
               const Group &     parentGroup,
               const bool        mpi,
               const Mode        mode)
    : HDF5Object(name, mpi)
  {
    hdf5_reference = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Gclose(*pointer);
      delete pointer;
    });
    switch (mode)
      {
        case (Mode::create):
          *hdf5_reference = H5Gcreate2(*(parentGroup.hdf5_reference),
                                       name.data(),
                                       H5P_DEFAULT,
                                       H5P_DEFAULT,
                                       H5P_DEFAULT);
          break;
        case (Mode::open):
          *hdf5_reference =
            H5Gopen2(*(parentGroup.hdf5_reference), name.data(), H5P_DEFAULT);
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
  }

  Group::Group(const std::string name, const bool mpi)
    : HDF5Object(name, mpi)
  {}

  Group
  Group::group(const std::string name)
  {
    return Group(name, *this, mpi, Mode::open);
  }

  Group
  Group::create_group(const std::string name)
  {
    return Group(name, *this, mpi, Mode::create);
  }

  DataSet
  Group::dataset(const std::string name)
  {
    return DataSet(name, *hdf5_reference, mpi);
  }

  template <typename number>
  DataSet
  Group::create_dataset(const std::string          name,
                        const std::vector<hsize_t> dimensions) const
  {
    std::shared_ptr<hid_t> t_type = internal::get_hdf5_datatype<number>();
    return DataSet(name, *hdf5_reference, dimensions, t_type, mpi);
  }

  template <typename number>
  void
  Group::write_dataset(const std::string          name,
                       const std::vector<number> &data) const
  {
    std::vector<hsize_t> dimensions = {data.size()};
    auto                 dataset    = create_dataset<number>(name, dimensions);
    dataset.write(data);
  }

  template <typename number>
  void
  Group::write_dataset(const std::string         name,
                       const FullMatrix<number> &data) const
  {
    std::vector<hsize_t> dimensions = {data.m(), data.n()};
    auto                 dataset    = create_dataset<number>(name, dimensions);
    dataset.write(data);
  }


  File::File(const std::string name,
             const bool        mpi,
             const MPI_Comm    mpi_communicator,
             const Mode        mode)
    : Group(name, mpi)
  {
    hdf5_reference = std::shared_ptr<hid_t>(new hid_t, [](auto pointer) {
      // Relase the HDF5 resource
      H5Fclose(*pointer);
      delete pointer;
    });

    hid_t          plist;
    const MPI_Info info = MPI_INFO_NULL;

    if (mpi)
      {
#  ifndef DEAL_II_WITH_MPI
        AssertThrow(false, ExcMessage("MPI support is disabled."));
#  endif // DEAL_II_WITH_MPI
#  ifndef H5_HAVE_PARALLEL
        AssertThrow(false, ExcMessage("HDF5 parallel support is disabled."));
#  endif // H5_HAVE_PARALLEL
        plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist, mpi_communicator, info);
      }
    else
      {
        plist = H5P_DEFAULT;
      }

    switch (mode)
      {
        case (Mode::create):
          *hdf5_reference =
            H5Fcreate(name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
          break;
        case (Mode::open):
          *hdf5_reference = H5Fopen(name.data(), H5F_ACC_RDWR, plist);
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }

    if (mpi)
      {
        // Relase the HDF5 resource
        H5Pclose(plist);
      }
  }

  File::File(const std::string name,
             const MPI_Comm    mpi_communicator,
             const Mode        mode)
    : File(name, true, mpi_communicator, mode)
  {}

  File::File(const std::string name, const Mode mode)
    : File(name, false, MPI_COMM_NULL, mode)
  {}

#  ifndef DOXYGEN

  // explicit instantiations of functions
  template float
  HDF5Object::attr<float>(const std::string attr_name) const;
  template double
  HDF5Object::attr<double>(const std::string attr_name) const;
  template std::complex<float>
  HDF5Object::attr<std::complex<float>>(const std::string attr_name) const;
  template std::complex<double>
  HDF5Object::attr<std::complex<double>>(const std::string attr_name) const;
  template int
  HDF5Object::attr<int>(const std::string attr_name) const;
  template unsigned int
  HDF5Object::attr<unsigned int>(const std::string attr_name) const;
  // The specialization of HDF5Object::attr<std::string> has been defined above

  template void
  HDF5Object::write_attr<float>(const std::string attr_name, float value) const;
  template void
  HDF5Object::write_attr<double>(const std::string attr_name,
                                 double            value) const;
  template void
  HDF5Object::write_attr<std::complex<float>>(const std::string   attr_name,
                                              std::complex<float> value) const;
  template void
  HDF5Object::write_attr<std::complex<double>>(
    const std::string    attr_name,
    std::complex<double> value) const;
  template void
  HDF5Object::write_attr<int>(const std::string attr_name, int value) const;
  template void
  HDF5Object::write_attr<unsigned int>(const std::string attr_name,
                                       unsigned int      value) const;

  template std::vector<float>
  DataSet::read<std::vector, float>();
  template std::vector<double>
  DataSet::read<std::vector, double>();
  template std::vector<std::complex<float>>
  DataSet::read<std::vector, std::complex<float>>();
  template std::vector<std::complex<double>>
  DataSet::read<std::vector, std::complex<double>>();
  template std::vector<int>
  DataSet::read<std::vector, int>();
  template std::vector<unsigned int>
  DataSet::read<std::vector, unsigned int>();
  template FullMatrix<float>
  DataSet::read<FullMatrix, float>();
  template FullMatrix<double>
  DataSet::read<FullMatrix, double>();
  template FullMatrix<std::complex<float>>
  DataSet::read<FullMatrix, std::complex<float>>();
  template FullMatrix<std::complex<double>>
  DataSet::read<FullMatrix, std::complex<double>>();

  template std::vector<float>
  DataSet::read_selection<float>(const std::vector<hsize_t> coordinates);
  template std::vector<double>
  DataSet::read_selection<double>(const std::vector<hsize_t> coordinates);
  template std::vector<std::complex<float>>
  DataSet::read_selection<std::complex<float>>(
    const std::vector<hsize_t> coordinates);
  template std::vector<std::complex<double>>
  DataSet::read_selection<std::complex<double>>(
    const std::vector<hsize_t> coordinates);
  template std::vector<int>
  DataSet::read_selection<int>(const std::vector<hsize_t> coordinates);
  template std::vector<unsigned int>
  DataSet::read_selection<unsigned int>(const std::vector<hsize_t> coordinates);

  template std::vector<float>
  DataSet::read_hyperslab<std::vector, float>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template std::vector<double>
  DataSet::read_hyperslab<std::vector, double>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template std::vector<std::complex<float>>
  DataSet::read_hyperslab<std::vector, std::complex<float>>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template std::vector<std::complex<double>>
  DataSet::read_hyperslab<std::vector, std::complex<double>>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template std::vector<int>
  DataSet::read_hyperslab<std::vector, int>(const std::vector<hsize_t> &offset,
                                            const std::vector<hsize_t> &count);
  template std::vector<unsigned int>
  DataSet::read_hyperslab<std::vector, unsigned int>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template FullMatrix<float>
  DataSet::read_hyperslab<FullMatrix, float>(const std::vector<hsize_t> &offset,
                                             const std::vector<hsize_t> &count);
  template FullMatrix<double>
  DataSet::read_hyperslab<FullMatrix, double>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template FullMatrix<std::complex<float>>
  DataSet::read_hyperslab<FullMatrix, std::complex<float>>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template FullMatrix<std::complex<double>>
  DataSet::read_hyperslab<FullMatrix, std::complex<double>>(
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);

  template void
  DataSet::read_none<float>();
  template void
  DataSet::read_none<double>();
  template void
  DataSet::read_none<std::complex<float>>();
  template void
  DataSet::read_none<std::complex<double>>();
  template void
  DataSet::read_none<int>();
  template void
  DataSet::read_none<unsigned int>();

  template void
  DataSet::write<std::vector, float>(const std::vector<float> &data);
  template void
  DataSet::write<std::vector, double>(const std::vector<double> &data);
  template void
  DataSet::write<std::vector, std::complex<float>>(
    const std::vector<std::complex<float>> &data);
  template void
  DataSet::write<std::vector, std::complex<double>>(
    const std::vector<std::complex<double>> &data);
  template void
  DataSet::write<std::vector, int>(const std::vector<int> &data);
  template void
  DataSet::write<std::vector, unsigned int>(
    const std::vector<unsigned int> &data);
  template void
  DataSet::write<FullMatrix, float>(const FullMatrix<float> &data);
  template void
  DataSet::write<FullMatrix, double>(const FullMatrix<double> &data);
  template void
  DataSet::write<FullMatrix, std::complex<float>>(
    const FullMatrix<std::complex<float>> &data);
  template void
  DataSet::write<FullMatrix, std::complex<double>>(
    const FullMatrix<std::complex<double>> &data);

  template void
  DataSet::write_selection<float>(const std::vector<float> & data,
                                  const std::vector<hsize_t> coordinates);
  template void
  DataSet::write_selection<double>(const std::vector<double> &data,
                                   const std::vector<hsize_t> coordinates);
  template void
  DataSet::write_selection<std::complex<float>>(
    const std::vector<std::complex<float>> &data,
    const std::vector<hsize_t>              coordinates);
  template void
  DataSet::write_selection<std::complex<double>>(
    const std::vector<std::complex<double>> &data,
    const std::vector<hsize_t>               coordinates);
  template void
  DataSet::write_selection<int>(const std::vector<int> &   data,
                                const std::vector<hsize_t> coordinates);
  template void
  DataSet::write_selection<unsigned int>(
    const std::vector<unsigned int> &data,
    const std::vector<hsize_t>       coordinates);

  template void
  DataSet::write_hyperslab<std::vector, float>(
    const std::vector<float> &  data,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template void
  DataSet::write_hyperslab<std::vector, double>(
    const std::vector<double> & data,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template void
  DataSet::write_hyperslab<std::vector, std::complex<float>>(
    const std::vector<std::complex<float>> &data,
    const std::vector<hsize_t> &            offset,
    const std::vector<hsize_t> &            count);
  template void
  DataSet::write_hyperslab<std::vector, std::complex<double>>(
    const std::vector<std::complex<double>> &data,
    const std::vector<hsize_t> &             offset,
    const std::vector<hsize_t> &             count);
  template void
  DataSet::write_hyperslab<std::vector, int>(const std::vector<int> &    data,
                                             const std::vector<hsize_t> &offset,
                                             const std::vector<hsize_t> &count);
  template void
  DataSet::write_hyperslab<std::vector, unsigned int>(
    const std::vector<unsigned int> &data,
    const std::vector<hsize_t> &     offset,
    const std::vector<hsize_t> &     count);
  template void
  DataSet::write_hyperslab<FullMatrix, float>(
    const FullMatrix<float> &   data,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template void
  DataSet::write_hyperslab<FullMatrix, double>(
    const FullMatrix<double> &  data,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &count);
  template void
  DataSet::write_hyperslab<FullMatrix, std::complex<float>>(
    const FullMatrix<std::complex<float>> &data,
    const std::vector<hsize_t> &           offset,
    const std::vector<hsize_t> &           count);
  template void
  DataSet::write_hyperslab<FullMatrix, std::complex<double>>(
    const FullMatrix<std::complex<double>> &data,
    const std::vector<hsize_t> &            offset,
    const std::vector<hsize_t> &            count);

  template void
  DataSet::write_hyperslab<std::vector, float>(
    const std::vector<float> &  data,
    const std::vector<hsize_t> &data_dimensions,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &stride,
    const std::vector<hsize_t> &count,
    const std::vector<hsize_t> &block);
  template void
  DataSet::write_hyperslab<std::vector, double>(
    const std::vector<double> & data,
    const std::vector<hsize_t> &data_dimensions,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &stride,
    const std::vector<hsize_t> &count,
    const std::vector<hsize_t> &block);
  template void
  DataSet::write_hyperslab<std::vector, std::complex<float>>(
    const std::vector<std::complex<float>> &data,
    const std::vector<hsize_t> &            data_dimensions,
    const std::vector<hsize_t> &            offset,
    const std::vector<hsize_t> &            stride,
    const std::vector<hsize_t> &            count,
    const std::vector<hsize_t> &            block);
  template void
  DataSet::write_hyperslab<std::vector, std::complex<double>>(
    const std::vector<std::complex<double>> &data,
    const std::vector<hsize_t> &             data_dimensions,
    const std::vector<hsize_t> &             offset,
    const std::vector<hsize_t> &             stride,
    const std::vector<hsize_t> &             count,
    const std::vector<hsize_t> &             block);
  template void
  DataSet::write_hyperslab<std::vector, int>(
    const std::vector<int> &    data,
    const std::vector<hsize_t> &data_dimensions,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &stride,
    const std::vector<hsize_t> &count,
    const std::vector<hsize_t> &block);
  template void
  DataSet::write_hyperslab<std::vector, unsigned int>(
    const std::vector<unsigned int> &data,
    const std::vector<hsize_t> &     data_dimensions,
    const std::vector<hsize_t> &     offset,
    const std::vector<hsize_t> &     stride,
    const std::vector<hsize_t> &     count,
    const std::vector<hsize_t> &     block);
  template void
  DataSet::write_hyperslab<FullMatrix, float>(
    const FullMatrix<float> &   data,
    const std::vector<hsize_t> &data_dimensions,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &stride,
    const std::vector<hsize_t> &count,
    const std::vector<hsize_t> &block);
  template void
  DataSet::write_hyperslab<FullMatrix, double>(
    const FullMatrix<double> &  data,
    const std::vector<hsize_t> &data_dimensions,
    const std::vector<hsize_t> &offset,
    const std::vector<hsize_t> &stride,
    const std::vector<hsize_t> &count,
    const std::vector<hsize_t> &block);
  template void
  DataSet::write_hyperslab<FullMatrix, std::complex<float>>(
    const FullMatrix<std::complex<float>> &data,
    const std::vector<hsize_t> &           data_dimensions,
    const std::vector<hsize_t> &           offset,
    const std::vector<hsize_t> &           stride,
    const std::vector<hsize_t> &           count,
    const std::vector<hsize_t> &           block);
  template void
  DataSet::write_hyperslab<FullMatrix, std::complex<double>>(
    const FullMatrix<std::complex<double>> &data,
    const std::vector<hsize_t> &            data_dimensions,
    const std::vector<hsize_t> &            offset,
    const std::vector<hsize_t> &            stride,
    const std::vector<hsize_t> &            count,
    const std::vector<hsize_t> &            block);

  template void
  DataSet::write_none<float>();
  template void
  DataSet::write_none<double>();
  template void
  DataSet::write_none<std::complex<float>>();
  template void
  DataSet::write_none<std::complex<double>>();
  template void
  DataSet::write_none<int>();
  template void
  DataSet::write_none<unsigned int>();

  template DataSet
  Group::create_dataset<float>(const std::string          name,
                               const std::vector<hsize_t> dimensions) const;
  template DataSet
  Group::create_dataset<double>(const std::string          name,
                                const std::vector<hsize_t> dimensions) const;
  template DataSet
  Group::create_dataset<std::complex<float>>(
    const std::string          name,
    const std::vector<hsize_t> dimensions) const;
  template DataSet
  Group::create_dataset<std::complex<double>>(
    const std::string          name,
    const std::vector<hsize_t> dimensions) const;
  template DataSet
  Group::create_dataset<int>(const std::string          name,
                             const std::vector<hsize_t> dimensions) const;
  template DataSet
  Group::create_dataset<unsigned int>(
    const std::string          name,
    const std::vector<hsize_t> dimensions) const;

  template void
  Group::write_dataset(const std::string         name,
                       const std::vector<float> &data) const;
  template void
  Group::write_dataset(const std::string          name,
                       const std::vector<double> &data) const;
  template void
  Group::write_dataset(const std::string                       name,
                       const std::vector<std::complex<float>> &data) const;
  template void
  Group::write_dataset(const std::string                        name,
                       const std::vector<std::complex<double>> &data) const;
  template void
  Group::write_dataset(const std::string       name,
                       const std::vector<int> &data) const;
  template void
  Group::write_dataset(const std::string                name,
                       const std::vector<unsigned int> &data) const;
  template void
  Group::write_dataset(const std::string        name,
                       const FullMatrix<float> &data) const;
  template void
  Group::write_dataset(const std::string         name,
                       const FullMatrix<double> &data) const;
  template void
  Group::write_dataset(const std::string                      name,
                       const FullMatrix<std::complex<float>> &data) const;
  template void
  Group::write_dataset(const std::string                       name,
                       const FullMatrix<std::complex<double>> &data) const;
#  endif // DOXYGEN

} // namespace HDF5

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_HDF5
