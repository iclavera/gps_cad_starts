FILE(REMOVE_RECURSE
  "msg_gen"
  "srv_gen"
  "msg_gen"
  "srv_gen"
  "src/gps_agent_pkg/msg"
  "src/gps_agent_pkg/srv"
  "CMakeFiles/ROSBUILD_genmsg_py"
  "src/gps_agent_pkg/msg/__init__.py"
  "src/gps_agent_pkg/msg/_SampleResult.py"
  "src/gps_agent_pkg/msg/_PositionCommand.py"
  "src/gps_agent_pkg/msg/_CaffeParams.py"
  "src/gps_agent_pkg/msg/_DataRequest.py"
  "src/gps_agent_pkg/msg/_LinGaussParams.py"
  "src/gps_agent_pkg/msg/_TfParams.py"
  "src/gps_agent_pkg/msg/_TrialCommand.py"
  "src/gps_agent_pkg/msg/_ProxyParams.py"
  "src/gps_agent_pkg/msg/_ControllerParams.py"
  "src/gps_agent_pkg/msg/_RelaxCommand.py"
  "src/gps_agent_pkg/msg/_TfActionCommand.py"
  "src/gps_agent_pkg/msg/_TfObsData.py"
  "src/gps_agent_pkg/msg/_DataType.py"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/ROSBUILD_genmsg_py.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
