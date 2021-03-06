/* Auto-generated by genmsg_cpp for file /home/michael/gps_cad/src/gps_agent_pkg/srv/ProxyControl.srv */
#ifndef GPS_AGENT_PKG_SERVICE_PROXYCONTROL_H
#define GPS_AGENT_PKG_SERVICE_PROXYCONTROL_H
#include <string>
#include <vector>
#include <map>
#include <ostream>
#include "ros/serialization.h"
#include "ros/builtin_message_traits.h"
#include "ros/message_operations.h"
#include "ros/time.h"

#include "ros/macros.h"

#include "ros/assert.h"

#include "ros/service_traits.h"




namespace gps_agent_pkg
{
template <class ContainerAllocator>
struct ProxyControlRequest_ {
  typedef ProxyControlRequest_<ContainerAllocator> Type;

  ProxyControlRequest_()
  : obs()
  {
  }

  ProxyControlRequest_(const ContainerAllocator& _alloc)
  : obs(_alloc)
  {
  }

  typedef std::vector<double, typename ContainerAllocator::template rebind<double>::other >  _obs_type;
  std::vector<double, typename ContainerAllocator::template rebind<double>::other >  obs;


  typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator>  const> ConstPtr;
}; // struct ProxyControlRequest
typedef  ::gps_agent_pkg::ProxyControlRequest_<std::allocator<void> > ProxyControlRequest;

typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlRequest> ProxyControlRequestPtr;
typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlRequest const> ProxyControlRequestConstPtr;



template <class ContainerAllocator>
struct ProxyControlResponse_ {
  typedef ProxyControlResponse_<ContainerAllocator> Type;

  ProxyControlResponse_()
  : action()
  {
  }

  ProxyControlResponse_(const ContainerAllocator& _alloc)
  : action(_alloc)
  {
  }

  typedef std::vector<double, typename ContainerAllocator::template rebind<double>::other >  _action_type;
  std::vector<double, typename ContainerAllocator::template rebind<double>::other >  action;


  typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator>  const> ConstPtr;
}; // struct ProxyControlResponse
typedef  ::gps_agent_pkg::ProxyControlResponse_<std::allocator<void> > ProxyControlResponse;

typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlResponse> ProxyControlResponsePtr;
typedef boost::shared_ptr< ::gps_agent_pkg::ProxyControlResponse const> ProxyControlResponseConstPtr;


struct ProxyControl
{

typedef ProxyControlRequest Request;
typedef ProxyControlResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;
}; // struct ProxyControl
} // namespace gps_agent_pkg

namespace ros
{
namespace message_traits
{
template<class ContainerAllocator> struct IsMessage< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > : public TrueType {};
template<class ContainerAllocator> struct IsMessage< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator>  const> : public TrueType {};
template<class ContainerAllocator>
struct MD5Sum< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > {
  static const char* value() 
  {
    return "8dac64abe4f5eba5d19614ccef1fe66c";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> &) { return value(); } 
  static const uint64_t static_value1 = 0x8dac64abe4f5eba5ULL;
  static const uint64_t static_value2 = 0xd19614ccef1fe66cULL;
};

template<class ContainerAllocator>
struct DataType< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > {
  static const char* value() 
  {
    return "gps_agent_pkg/ProxyControlRequest";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct Definition< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > {
  static const char* value() 
  {
    return "float64[] obs\n\
\n\
";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> &) { return value(); } 
};

} // namespace message_traits
} // namespace ros


namespace ros
{
namespace message_traits
{
template<class ContainerAllocator> struct IsMessage< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > : public TrueType {};
template<class ContainerAllocator> struct IsMessage< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator>  const> : public TrueType {};
template<class ContainerAllocator>
struct MD5Sum< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > {
  static const char* value() 
  {
    return "79f44d272f2ebe04451185b0dea57684";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> &) { return value(); } 
  static const uint64_t static_value1 = 0x79f44d272f2ebe04ULL;
  static const uint64_t static_value2 = 0x451185b0dea57684ULL;
};

template<class ContainerAllocator>
struct DataType< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > {
  static const char* value() 
  {
    return "gps_agent_pkg/ProxyControlResponse";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct Definition< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > {
  static const char* value() 
  {
    return "float64[] action\n\
\n\
\n\
";
  }

  static const char* value(const  ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> &) { return value(); } 
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

template<class ContainerAllocator> struct Serializer< ::gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> >
{
  template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
  {
    stream.next(m.obs);
  }

  ROS_DECLARE_ALLINONE_SERIALIZER;
}; // struct ProxyControlRequest_
} // namespace serialization
} // namespace ros


namespace ros
{
namespace serialization
{

template<class ContainerAllocator> struct Serializer< ::gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> >
{
  template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
  {
    stream.next(m.action);
  }

  ROS_DECLARE_ALLINONE_SERIALIZER;
}; // struct ProxyControlResponse_
} // namespace serialization
} // namespace ros

namespace ros
{
namespace service_traits
{
template<>
struct MD5Sum<gps_agent_pkg::ProxyControl> {
  static const char* value() 
  {
    return "df8911a3dc99abf752eaebab6600df26";
  }

  static const char* value(const gps_agent_pkg::ProxyControl&) { return value(); } 
};

template<>
struct DataType<gps_agent_pkg::ProxyControl> {
  static const char* value() 
  {
    return "gps_agent_pkg/ProxyControl";
  }

  static const char* value(const gps_agent_pkg::ProxyControl&) { return value(); } 
};

template<class ContainerAllocator>
struct MD5Sum<gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > {
  static const char* value() 
  {
    return "df8911a3dc99abf752eaebab6600df26";
  }

  static const char* value(const gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct DataType<gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> > {
  static const char* value() 
  {
    return "gps_agent_pkg/ProxyControl";
  }

  static const char* value(const gps_agent_pkg::ProxyControlRequest_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct MD5Sum<gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > {
  static const char* value() 
  {
    return "df8911a3dc99abf752eaebab6600df26";
  }

  static const char* value(const gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> &) { return value(); } 
};

template<class ContainerAllocator>
struct DataType<gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> > {
  static const char* value() 
  {
    return "gps_agent_pkg/ProxyControl";
  }

  static const char* value(const gps_agent_pkg::ProxyControlResponse_<ContainerAllocator> &) { return value(); } 
};

} // namespace service_traits
} // namespace ros

#endif // GPS_AGENT_PKG_SERVICE_PROXYCONTROL_H

