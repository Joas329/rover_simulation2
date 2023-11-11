#ifndef YURS_VELOCITY_CONTROLLERS__VISIBILITY_CONTROL_H_
#define YURS_VELOCITY_CONTROLLERS__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define YURS_VELOCITY_CONTROLLERS_EXPORT __attribute__((dllexport))
#define YURS_VELOCITY_CONTROLLERS_IMPORT __attribute__((dllimport))
#else
#define YURS_VELOCITY_CONTROLLERS_EXPORT __declspec(dllexport)
#define YURS_VELOCITY_CONTROLLERS_IMPORT __declspec(dllimport)
#endif
#ifdef YURS_VELOCITY_CONTROLLERS_BUILDING_DLL
#define YURS_VELOCITY_CONTROLLERS_PUBLIC YURS_VELOCITY_CONTROLLERS_EXPORT
#else
#define YURS_VELOCITY_CONTROLLERS_PUBLIC YURS_VELOCITY_CONTROLLERS_IMPORT
#endif
#define YURS_VELOCITY_CONTROLLERS_PUBLIC_TYPE YURS_VELOCITY_CONTROLLERS_PUBLIC
#define YURS_VELOCITY_CONTROLLERS_LOCAL
#else
#define YURS_VELOCITY_CONTROLLERS_EXPORT __attribute__((visibility("default")))
#define YURS_VELOCITY_CONTROLLERS_IMPORT
#if __GNUC__ >= 4
#define YURS_VELOCITY_CONTROLLERS_PUBLIC __attribute__((visibility("default")))
#define YURS_VELOCITY_CONTROLLERS_LOCAL __attribute__((visibility("hidden")))
#else
#define YURS_VELOCITY_CONTROLLERS_PUBLIC
#define YURS_VELOCITY_CONTROLLERS_LOCAL
#endif
#define YURS_VELOCITY_CONTROLLERS_PUBLIC_TYPE
#endif

#endif  // YURS_VELOCITY_CONTROLLERS__VISIBILITY_CONTROL_H_