FILE(REMOVE_RECURSE
  "msg_gen"
  "srv_gen"
  "msg_gen"
  "srv_gen"
  "src/gps_agent_pkg/msg"
  "src/gps_agent_pkg/srv"
  "CMakeFiles/ROSBUILD_genmsg_cpp"
  "msg_gen/cpp/include/gps_agent_pkg/SampleResult.h"
  "msg_gen/cpp/include/gps_agent_pkg/PositionCommand.h"
  "msg_gen/cpp/include/gps_agent_pkg/CaffeParams.h"
  "msg_gen/cpp/include/gps_agent_pkg/DataRequest.h"
  "msg_gen/cpp/include/gps_agent_pkg/LinGaussParams.h"
  "msg_gen/cpp/include/gps_agent_pkg/TfParams.h"
  "msg_gen/cpp/include/gps_agent_pkg/TrialCommand.h"
  "msg_gen/cpp/include/gps_agent_pkg/ProxyParams.h"
  "msg_gen/cpp/include/gps_agent_pkg/ControllerParams.h"
  "msg_gen/cpp/include/gps_agent_pkg/RelaxCommand.h"
  "msg_gen/cpp/include/gps_agent_pkg/TfActionCommand.h"
  "msg_gen/cpp/include/gps_agent_pkg/TfObsData.h"
  "msg_gen/cpp/include/gps_agent_pkg/DataType.h"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_cpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
