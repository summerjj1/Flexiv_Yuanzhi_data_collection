# generated from rosidl_generator_py/resource/_idl.py.em
# with input from arx5_arm_msg:msg/RobotStatus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100
import math  # noqa: E402, I100

# Member 'end_pos'
# Member 'joint_pos'
# Member 'joint_vel'
# Member 'joint_cur'
import numpy  # noqa: E402, I100
import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotStatus(type):
    """Metaclass of message 'RobotStatus'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {}

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support

            module = import_type_support("arx5_arm_msg")
        except ImportError:
            import logging
            import traceback

            logger = logging.getLogger("arx5_arm_msg.msg.RobotStatus")
            logger.debug(
                "Failed to import needed modules for type support:\n"
                + traceback.format_exc()
            )
        else:
            cls._CREATE_ROS_MESSAGE = (
                module.create_ros_message_msg__msg__robot_status
            )
            cls._CONVERT_FROM_PY = (
                module.convert_from_py_msg__msg__robot_status
            )
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_status
            cls._DESTROY_ROS_MESSAGE = (
                module.destroy_ros_message_msg__msg__robot_status
            )

            from std_msgs.msg import Header

            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {}


class RobotStatus(metaclass=Metaclass_RobotStatus):
    """Message class 'RobotStatus'."""

    __slots__ = [
        "_header",
        "_end_pos",
        "_joint_pos",
        "_joint_vel",
        "_joint_cur",
    ]

    _fields_and_field_types = {
        "header": "std_msgs/Header",
        "end_pos": "double[6]",
        "joint_pos": "double[7]",
        "joint_vel": "double[7]",
        "joint_cur": "double[7]",
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(
            ["std_msgs", "msg"], "Header"
        ),  # noqa: E501
        rosidl_parser.definition.Array(
            rosidl_parser.definition.BasicType("double"), 6
        ),  # noqa: E501
        rosidl_parser.definition.Array(
            rosidl_parser.definition.BasicType("double"), 7
        ),  # noqa: E501
        rosidl_parser.definition.Array(
            rosidl_parser.definition.BasicType("double"), 7
        ),  # noqa: E501
        rosidl_parser.definition.Array(
            rosidl_parser.definition.BasicType("double"), 7
        ),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all(
            "_" + key in self.__slots__ for key in kwargs.keys()
        ), "Invalid arguments passed to constructor: %s" % ", ".join(
            sorted(k for k in kwargs.keys() if "_" + k not in self.__slots__)
        )
        from std_msgs.msg import Header

        self.header = kwargs.get("header", Header())
        if "end_pos" not in kwargs:
            self.end_pos = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.end_pos = numpy.array(
                kwargs.get("end_pos"), dtype=numpy.float64
            )
            assert self.end_pos.shape == (6,)
        if "joint_pos" not in kwargs:
            self.joint_pos = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.joint_pos = numpy.array(
                kwargs.get("joint_pos"), dtype=numpy.float64
            )
            assert self.joint_pos.shape == (7,)
        if "joint_vel" not in kwargs:
            self.joint_vel = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.joint_vel = numpy.array(
                kwargs.get("joint_vel"), dtype=numpy.float64
            )
            assert self.joint_vel.shape == (7,)
        if "joint_cur" not in kwargs:
            self.joint_cur = numpy.zeros(7, dtype=numpy.float64)
        else:
            self.joint_cur = numpy.array(
                kwargs.get("joint_cur"), dtype=numpy.float64
            )
            assert self.joint_cur.shape == (7,)

    def __repr__(self):
        typename = self.__class__.__module__.split(".")
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence)
                and isinstance(
                    t.value_type, rosidl_parser.definition.BasicType
                )
                and t.value_type.typename
                in [
                    "float",
                    "double",
                    "int8",
                    "uint8",
                    "int16",
                    "uint16",
                    "int32",
                    "uint32",
                    "int64",
                    "uint64",
                ]
            ):
                if len(field) == 0:
                    fieldstr = "[]"
                else:
                    assert fieldstr.startswith("array(")
                    prefix = "array('X', "
                    suffix = ")"
                    fieldstr = fieldstr[len(prefix) : -len(suffix)]
            args.append(s[1:] + "=" + fieldstr)
        return "%s(%s)" % (".".join(typename), ", ".join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if all(self.end_pos != other.end_pos):
            return False
        if all(self.joint_pos != other.joint_pos):
            return False
        if all(self.joint_vel != other.joint_vel):
            return False
        if all(self.joint_cur != other.joint_cur):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy

        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header

            assert isinstance(
                value, Header
            ), "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def end_pos(self):
        """Message field 'end_pos'."""
        return self._end_pos

    @end_pos.setter
    def end_pos(self, value):
        if isinstance(value, numpy.ndarray):
            assert (
                value.dtype == numpy.float64
            ), "The 'end_pos' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert (
                value.size == 6
            ), "The 'end_pos' numpy.ndarray() must have a size of 6"
            self._end_pos = value
            return
        if __debug__:
            from collections import UserList, UserString
            from collections.abc import Sequence, Set

            assert (
                (
                    isinstance(value, Sequence)
                    or isinstance(value, Set)
                    or isinstance(value, UserList)
                )
                and not isinstance(value, str)
                and not isinstance(value, UserString)
                and len(value) == 6
                and all(isinstance(v, float) for v in value)
                and all(
                    not (
                        val < -1.7976931348623157e308
                        or val > 1.7976931348623157e308
                    )
                    or math.isinf(val)
                    for val in value
                )
            ), "The 'end_pos' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._end_pos = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joint_pos(self):
        """Message field 'joint_pos'."""
        return self._joint_pos

    @joint_pos.setter
    def joint_pos(self, value):
        if isinstance(value, numpy.ndarray):
            assert (
                value.dtype == numpy.float64
            ), "The 'joint_pos' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert (
                value.size == 7
            ), "The 'joint_pos' numpy.ndarray() must have a size of 7"
            self._joint_pos = value
            return
        if __debug__:
            from collections import UserList, UserString
            from collections.abc import Sequence, Set

            assert (
                (
                    isinstance(value, Sequence)
                    or isinstance(value, Set)
                    or isinstance(value, UserList)
                )
                and not isinstance(value, str)
                and not isinstance(value, UserString)
                and len(value) == 7
                and all(isinstance(v, float) for v in value)
                and all(
                    not (
                        val < -1.7976931348623157e308
                        or val > 1.7976931348623157e308
                    )
                    or math.isinf(val)
                    for val in value
                )
            ), "The 'joint_pos' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_pos = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joint_vel(self):
        """Message field 'joint_vel'."""
        return self._joint_vel

    @joint_vel.setter
    def joint_vel(self, value):
        if isinstance(value, numpy.ndarray):
            assert (
                value.dtype == numpy.float64
            ), "The 'joint_vel' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert (
                value.size == 7
            ), "The 'joint_vel' numpy.ndarray() must have a size of 7"
            self._joint_vel = value
            return
        if __debug__:
            from collections import UserList, UserString
            from collections.abc import Sequence, Set

            assert (
                (
                    isinstance(value, Sequence)
                    or isinstance(value, Set)
                    or isinstance(value, UserList)
                )
                and not isinstance(value, str)
                and not isinstance(value, UserString)
                and len(value) == 7
                and all(isinstance(v, float) for v in value)
                and all(
                    not (
                        val < -1.7976931348623157e308
                        or val > 1.7976931348623157e308
                    )
                    or math.isinf(val)
                    for val in value
                )
            ), "The 'joint_vel' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_vel = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joint_cur(self):
        """Message field 'joint_cur'."""
        return self._joint_cur

    @joint_cur.setter
    def joint_cur(self, value):
        if isinstance(value, numpy.ndarray):
            assert (
                value.dtype == numpy.float64
            ), "The 'joint_cur' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert (
                value.size == 7
            ), "The 'joint_cur' numpy.ndarray() must have a size of 7"
            self._joint_cur = value
            return
        if __debug__:
            from collections import UserList, UserString
            from collections.abc import Sequence, Set

            assert (
                (
                    isinstance(value, Sequence)
                    or isinstance(value, Set)
                    or isinstance(value, UserList)
                )
                and not isinstance(value, str)
                and not isinstance(value, UserString)
                and len(value) == 7
                and all(isinstance(v, float) for v in value)
                and all(
                    not (
                        val < -1.7976931348623157e308
                        or val > 1.7976931348623157e308
                    )
                    or math.isinf(val)
                    for val in value
                )
            ), "The 'joint_cur' field must be a set or sequence with length 7 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_cur = numpy.array(value, dtype=numpy.float64)
