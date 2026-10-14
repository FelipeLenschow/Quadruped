#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quadruped_control/controller.hpp"
#include "quadruped_control/estimator.hpp"
#include "quadruped_control/kinematics.hpp"
#include "quadruped_control/policy.hpp"

namespace py = pybind11;
using namespace quadruped_control;

PYBIND11_MODULE(_core, m) {
  py::class_<LegGeometry>(m, "LegGeometry")
      .def(py::init<>())
      .def_static("from_yaml", &LegGeometry::from_yaml, py::arg("path"))
      .def_readwrite("hip_origins", &LegGeometry::hip_origins)
      .def_readwrite("thigh_offset_y", &LegGeometry::thigh_offset_y)
      .def_readwrite("thigh_length", &LegGeometry::thigh_length)
      .def_readwrite("calf_length", &LegGeometry::calf_length)
      .def_readwrite("foot_x_offset", &LegGeometry::foot_x_offset);

  py::class_<Kinematics>(m, "Kinematics")
      .def(py::init<const LegGeometry&>(), py::arg("geometry"))
      .def("foot_position_body", &Kinematics::foot_position_body, py::arg("leg_idx"), py::arg("q_leg"))
      .def("foot_jacobian_body", &Kinematics::foot_jacobian_body, py::arg("leg_idx"), py::arg("q_leg"))
      .def("leg_chain_body", &Kinematics::leg_chain_body, py::arg("leg_idx"), py::arg("q_leg"))
      .def_property_readonly("geometry", &Kinematics::geometry);

  py::class_<EstimatorParams>(m, "EstimatorParams")
      .def(py::init([](double dt, double q_vel, double q_bias, double r_meas, double max_speed,
                       double no_contact_relax_time) {
             return EstimatorParams{dt, q_vel, q_bias, r_meas, max_speed, no_contact_relax_time};
           }),
           py::arg("dt") = 0.02, py::arg("q_vel") = 0.01, py::arg("q_bias") = 1e-3,
           py::arg("r_meas") = 0.05, py::arg("max_speed") = 5.0,
           py::arg("no_contact_relax_time") = 0.3)
      .def_readwrite("dt", &EstimatorParams::dt)
      .def_readwrite("q_vel", &EstimatorParams::q_vel)
      .def_readwrite("q_bias", &EstimatorParams::q_bias)
      .def_readwrite("r_meas", &EstimatorParams::r_meas)
      .def_readwrite("max_speed", &EstimatorParams::max_speed)
      .def_readwrite("no_contact_relax_time", &EstimatorParams::no_contact_relax_time);

  py::class_<StateEstimator>(m, "StateEstimator")
      .def(py::init<const Kinematics&, const EstimatorParams&>(), py::arg("kinematics"),
           py::arg("params") = EstimatorParams{})
      .def("reset", &StateEstimator::reset)
      .def("update", &StateEstimator::update, py::arg("quat_wxyz"), py::arg("accel_body"),
           py::arg("feet_contact"), py::arg("joint_pos") = py::none(),
           py::arg("joint_vel") = py::none(), py::arg("gyro_body") = py::none())
      .def_property_readonly("velocity", &StateEstimator::velocity)
      .def_property_readonly("accel_bias", &StateEstimator::accel_bias)
      .def_property_readonly("covariance", &StateEstimator::covariance)
      .def_property_readonly("params", &StateEstimator::params);

  py::class_<PolicySpec>(m, "PolicySpec")
      .def_static("from_file", &PolicySpec::from_file, py::arg("path"))
      .def_readonly("name", &PolicySpec::name)
      .def_readonly("source", &PolicySpec::source)
      .def_property_readonly("layout", [](const PolicySpec& s) { return to_string(s.layout); })
      .def_readonly("obs_dim", &PolicySpec::obs_dim)
      .def_readonly("obs_dim_single", &PolicySpec::obs_dim_single)
      .def_readonly("history_len", &PolicySpec::history_len)
      .def_readonly("action_scale", &PolicySpec::action_scale)
      .def_readonly("desired_qpos", &PolicySpec::desired_qpos);

  py::class_<PolicyRunner>(m, "PolicyRunner")
      .def(py::init<PolicySpec>(), py::arg("spec"))
      .def_static("from_file", [](const std::string& path) { return PolicyRunner(PolicySpec::from_file(path)); },
                  py::arg("path"))
      .def(
          "infer",
          [](PolicyRunner& r, const Vec4& quat, const Vec3& lin_vel, const Vec3& gyro, const Vec12& q,
             const Vec12& dq, const Eigen::VectorXd& commands, double dt) {
            return r.infer(ObsInput{quat, lin_vel, gyro, q, dq}, commands, dt);
          },
          py::arg("quat_wxyz"), py::arg("lin_vel"), py::arg("gyro"), py::arg("q"), py::arg("dq"),
          py::arg("commands"), py::arg("dt") = 0.02)
      .def("reset_history", &PolicyRunner::reset_history)
      .def("set_gait_vector", &PolicyRunner::set_gait_vector, py::arg("values"))
      .def(
          "set_gait",
          [](PolicyRunner& r, std::optional<std::string> gait, py::kwargs fields) {
            std::map<std::string, float> f;
            for (const auto& item : fields) f[item.first.cast<std::string>()] = item.second.cast<float>();
            r.set_gait(gait, f);
          },
          py::arg("gait") = py::none())
      .def_property_readonly("spec", &PolicyRunner::spec)
      .def_property_readonly("last_actions", &PolicyRunner::last_actions)
      .def_property_readonly("last_obs", &PolicyRunner::last_obs)
      .def_property_readonly("gait_command", &PolicyRunner::gait_command);

  py::enum_<LogLevel>(m, "LogLevel").value("INFO", LogLevel::Info).value("WARN", LogLevel::Warn).value("ERROR", LogLevel::Error);
  py::enum_<EventKind>(m, "EventKind").value("LOG", EventKind::Log).value("SAFETY_TRIP", EventKind::SafetyTrip);
  py::class_<Event>(m, "Event")
      .def_readonly("kind", &Event::kind)
      .def_readonly("level", &Event::level)
      .def_readonly("text", &Event::text);
  py::enum_<PolicyRequest>(m, "PolicyRequest")
      .value("REFUSED", PolicyRequest::Refused)
      .value("ALREADY_ACTIVE", PolicyRequest::AlreadyActive)
      .value("UNKNOWN", PolicyRequest::Unknown)
      .value("READY", PolicyRequest::Ready)
      .value("NEEDS_LOAD", PolicyRequest::NeedsLoad);

  py::class_<ControllerConfig>(m, "ControllerConfig")
      .def(py::init<>())
      .def_static("from_yaml", &ControllerConfig::from_yaml, py::arg("config_path"), py::arg("geometry_path"),
                  py::arg("sim_dt"), py::arg("use_estimator"), py::arg("safety_switch") = true)
      .def_readwrite("decimation", &ControllerConfig::decimation)
      .def_readwrite("sim_dt", &ControllerConfig::sim_dt)
      .def_readwrite("policy_dt", &ControllerConfig::policy_dt)
      .def_readwrite("action_scale", &ControllerConfig::action_scale)
      .def_readwrite("saturation", &ControllerConfig::saturation)
      .def_readwrite("hard_min", &ControllerConfig::hard_min)
      .def_readwrite("hard_max", &ControllerConfig::hard_max)
      .def_readwrite("joint_names", &ControllerConfig::joint_names)
      .def_readwrite("motor_max_torque", &ControllerConfig::motor_max_torque)
      .def_readwrite("watchdog_timeout", &ControllerConfig::watchdog_timeout)
      .def_readwrite("safety_policy_timeout", &ControllerConfig::safety_policy_timeout)
      .def_readwrite("lying_hold", &ControllerConfig::lying_hold)
      .def_readwrite("lying_tilt_deg", &ControllerConfig::lying_tilt_deg)
      .def_readwrite("lying_pose_err", &ControllerConfig::lying_pose_err)
      .def_readwrite("lying_joint_vel", &ControllerConfig::lying_joint_vel)
      .def_readwrite("lie_pose", &ControllerConfig::lie_pose)
      .def_readwrite("poses", &ControllerConfig::poses)
      .def_readwrite("pose_interp_duration", &ControllerConfig::pose_interp_duration)
      .def_readwrite("policy_switch_blend", &ControllerConfig::policy_switch_blend)
      .def_readwrite("mode_transition_duration", &ControllerConfig::mode_transition_duration)
      .def_readwrite("lin_accel", &ControllerConfig::lin_accel)
      .def_readwrite("lin_decel", &ControllerConfig::lin_decel)
      .def_readwrite("yaw_accel", &ControllerConfig::yaw_accel)
      .def_readwrite("yaw_decel", &ControllerConfig::yaw_decel)
      .def_readwrite("cmd_timeout", &ControllerConfig::cmd_timeout)
      .def_readwrite("use_estimator", &ControllerConfig::use_estimator)
      .def_readwrite("safety_switch", &ControllerConfig::safety_switch)
      .def_readwrite("estimator", &ControllerConfig::estimator)
      .def_readwrite("geometry", &ControllerConfig::geometry);

  py::class_<PolicyEntry>(m, "PolicyEntry")
      .def(py::init([](std::optional<std::vector<std::string>> protections, py::object commands, double clock_hz,
                       int switches) {
             PolicyEntry e;
             e.protections = protections ? protections_from_names(*protections) : static_cast<uint8_t>(kAllProtections);
             if (py::isinstance<py::str>(commands)) {
               if (commands.cast<std::string>() != "trick") throw std::invalid_argument("commands must be true, false or \"trick\"");
               e.commands = CommandMode::Trick;
             } else {
               e.commands = commands.cast<bool>() ? CommandMode::Velocity : CommandMode::Zero;
             }
             e.clock_hz = clock_hz;
             e.switches = switches;
             return e;
           }),
           py::arg("protections") = py::none(), py::arg("commands") = true, py::arg("clock_hz") = 0.0,
           py::arg("switches") = 0)
      .def_property_readonly("protections", [](const PolicyEntry& e) { return protection_names(e.protections); })
      .def_property_readonly("commands", [](const PolicyEntry& e) -> py::object {
        if (e.commands == CommandMode::Trick) return py::str("trick");
        return py::bool_(e.commands == CommandMode::Velocity);
      })
      .def_readonly("clock_hz", &PolicyEntry::clock_hz)
      .def_readonly("switches", &PolicyEntry::switches);

  py::class_<State>(m, "State")
      .def_readonly("quat_wxyz", &State::quat_wxyz)
      .def_readonly("gyro", &State::gyro)
      .def_readonly("accel", &State::accel)
      .def_readonly("q", &State::q)
      .def_readonly("dq", &State::dq)
      .def_readonly("base_pos", &State::base_pos)
      .def_readonly("has_pos", &State::has_pos)
      .def_readonly("feet_contact", &State::feet_contact)
      .def_readonly("base_lin_vel", &State::base_lin_vel)
      .def_readonly("base_lin_vel_sim", &State::base_lin_vel_sim)
      .def_readonly("base_lin_vel_est", &State::base_lin_vel_est);

  py::class_<StepResult>(m, "StepResult")
      .def_readonly("targets", &StepResult::targets)
      .def_readonly("max_torque", &StepResult::max_torque)
      .def_readonly("policy_step", &StepResult::policy_step)
      .def_readonly("pose_status", &StepResult::pose_status);

  py::class_<SafetyProcessor>(m, "SafetyProcessor")
      .def("heartbeat", &SafetyProcessor::heartbeat, py::arg("now"))
      .def("set_max_torque_percent", &SafetyProcessor::set_max_torque_percent)
      .def("set_base_tilt_limit", &SafetyProcessor::set_base_tilt_limit)
      .def("set_forward_tilt_limit", &SafetyProcessor::set_forward_tilt_limit)
      .def("set_rom_margin", &SafetyProcessor::set_rom_margin)
      .def("set_watchdog_timeout", &SafetyProcessor::set_watchdog_timeout)
      .def("estop", &SafetyProcessor::estop)
      .def("release_estop", &SafetyProcessor::release_estop)
      .def("set_numb", &SafetyProcessor::set_numb)
      .def("numb", &SafetyProcessor::numb)
      .def("release", &SafetyProcessor::release)
      .def("safety_reset", &SafetyProcessor::safety_reset)
      .def("clear_block", &SafetyProcessor::clear_block, py::arg("clear_reason") = true)
      .def("pin_rom_margin", &SafetyProcessor::pin_rom_margin, py::arg("margin"))
      .def("heartbeat_ok", &SafetyProcessor::heartbeat_ok, py::arg("now"))
      .def_property("active_max_torque", &SafetyProcessor::active_max_torque, &SafetyProcessor::set_active_max_torque)
      .def_property_readonly("is_estopped", &SafetyProcessor::is_estopped)
      .def_property_readonly("is_numb", &SafetyProcessor::is_numb)
      .def_property_readonly("is_policy_blocked", &SafetyProcessor::is_policy_blocked)
      .def_property_readonly("has_received_heartbeat", &SafetyProcessor::has_received_heartbeat)
      .def_property_readonly("last_heartbeat_time", &SafetyProcessor::last_heartbeat_time)
      .def_property_readonly("watchdog_timeout", &SafetyProcessor::watchdog_timeout)
      .def_property_readonly("global_max_torque", &SafetyProcessor::global_max_torque)
      .def_property_readonly("max_torque_percent", &SafetyProcessor::max_torque_percent)
      .def_property_readonly("rom_safety_margin", &SafetyProcessor::rom_margin)
      .def_property_readonly("trip_reason", &SafetyProcessor::trip_reason)
      .def_property_readonly("numb_reason", &SafetyProcessor::numb_reason)
      .def_property_readonly("soft_min", &SafetyProcessor::soft_min)
      .def_property_readonly("soft_max", &SafetyProcessor::soft_max)
      .def_property_readonly("joint_safe_min", &SafetyProcessor::joint_safe_min)
      .def_property_readonly("joint_safe_max", &SafetyProcessor::joint_safe_max);

  py::class_<PoseGenerator>(m, "PoseGenerator")
      .def_property_readonly("is_active", &PoseGenerator::is_active)
      .def_property_readonly("is_complete", &PoseGenerator::is_complete)
      .def_property_readonly("current_pose_name", &PoseGenerator::current_pose_name)
      .def_property_readonly("status", &PoseGenerator::status)
      .def_property_readonly("interp_duration", &PoseGenerator::interp_duration);

  py::class_<Controller>(m, "Controller")
      .def(py::init<ControllerConfig>(), py::arg("config"))
      .def("set_catalog", &Controller::set_catalog, py::arg("catalog"))
      .def("set_main_alias", &Controller::set_main_alias, py::arg("name"))
      .def("add_policy", py::overload_cast<const std::string&, PolicySpec, const PolicyEntry&>(&Controller::add_policy),
           py::arg("name"), py::arg("spec"), py::arg("entry"))
      .def("unload_policy", &Controller::unload_policy, py::arg("name"))
      .def("set_load_error", &Controller::set_load_error, py::arg("name"), py::arg("error"))
      .def("set_safety_policy", &Controller::set_safety_policy, py::arg("name"))
      .def("set_active_policy", &Controller::set_active_policy, py::arg("name"))
      .def("request_policy", &Controller::request_policy, py::arg("name"))
      .def("set_mode", &Controller::set_mode, py::arg("mode"))
      .def("force_mode", &Controller::force_mode, py::arg("mode"))
      .def("pose_command", &Controller::pose_command, py::arg("name"))
      .def("set_pose_interp_duration", &Controller::set_pose_interp_duration, py::arg("seconds"))
      .def("cmd_vel", &Controller::cmd_vel, py::arg("vx"), py::arg("vy"), py::arg("wz"), py::arg("now"))
      .def("trick", &Controller::trick, py::arg("values"), py::arg("now"))
      .def("gait", &Controller::gait, py::arg("values"))
      .def("gait_override", &Controller::gait_override, py::arg("values"), py::arg("now"))
      .def(
          "step",
          [](Controller& c, const Vec12& q, const Vec12& dq, const Vec4& quat, const Vec3& gyro, std::optional<Vec3> accel,
             std::optional<Vec3> pos, std::optional<Vec3> vel, std::optional<Vec4> contact,
             std::optional<Eigen::VectorXd> cmd_vel, double sim_time, double now) {
            RawState raw{q, dq, quat, gyro, accel, pos, vel, contact};
            return c.step(raw, cmd_vel, sim_time, now);
          },
          py::arg("q"), py::arg("dq"), py::arg("quat"), py::arg("gyro"), py::arg("accel") = py::none(),
          py::arg("pos") = py::none(), py::arg("vel") = py::none(), py::arg("contact") = py::none(), py::kw_only(),
          py::arg("cmd_vel") = py::none(), py::arg("sim_time"), py::arg("now"))
      .def("reset", &Controller::reset)
      .def("drain_events", &Controller::drain_events)
      .def_property_readonly("safety", py::overload_cast<>(&Controller::safety), py::return_value_policy::reference_internal)
      .def_property_readonly("pose", &Controller::pose, py::return_value_policy::reference_internal)
      .def_property_readonly("estimator", &Controller::estimator, py::return_value_policy::reference_internal)
      .def_property_readonly("state", &Controller::state)
      .def_property_readonly("latest_targets", &Controller::latest_targets)
      .def_property_readonly("mode", &Controller::mode)
      .def_property("transition_active", &Controller::transition_active, &Controller::set_transition_active)
      .def_property_readonly("active_policy", &Controller::active_policy)
      .def_property_readonly("pending_policy", &Controller::pending_policy)
      .def_property_readonly("policy_error", &Controller::policy_error)
      .def_property_readonly("safety_policy", &Controller::safety_policy)
      .def_property_readonly("safety_policy_active", &Controller::safety_policy_active)
      .def_property_readonly("available", &Controller::available)
      .def_property_readonly("loaded", &Controller::loaded)
      .def("has_policy", &Controller::has_policy, py::arg("name"))
      .def("runner", &Controller::runner, py::arg("name"), py::return_value_policy::reference_internal)
      .def("entry", &Controller::entry, py::arg("name"))
      .def("current_cmd_vel", &Controller::current_cmd_vel, py::arg("now"))
      .def("cmd_age", &Controller::cmd_age, py::arg("now"))
      .def("trick_command", &Controller::trick_command, py::arg("now"))
      .def_property_readonly("gait_override_active", &Controller::gait_override_active)
      .def_property_readonly("gait_command", [](const Controller& c) -> std::optional<VecXf> {
        const VecXf* g = c.gait_command();
        return g ? std::optional<VecXf>(*g) : std::nullopt;
      })
      .def_property("slew_enabled", [](const Controller& c) { return c.slew().enabled(); }, &Controller::set_slew_enabled)
      .def_property_readonly("last_command", &Controller::last_command)
      .def_property("decimation", &Controller::decimation, &Controller::set_decimation)
      .def_property("policy_dt", &Controller::policy_dt, &Controller::set_policy_dt);

  m.def("default_stance", &default_stance);

  m.def("rot_from_quat", &rot_from_quat, py::arg("quat_wxyz"));
  m.def("projected_gravity_b", &projected_gravity_b, py::arg("quat_wxyz"));
}
