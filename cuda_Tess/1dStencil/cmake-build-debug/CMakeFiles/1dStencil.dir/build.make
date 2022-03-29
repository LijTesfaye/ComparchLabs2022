# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tmp/tmp.UVC05cgjNz

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/tmp.UVC05cgjNz/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/1dStencil.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/1dStencil.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/1dStencil.dir/flags.make

CMakeFiles/1dStencil.dir/main.cu.o: CMakeFiles/1dStencil.dir/flags.make
CMakeFiles/1dStencil.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.UVC05cgjNz/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/1dStencil.dir/main.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /tmp/tmp.UVC05cgjNz/main.cu -o CMakeFiles/1dStencil.dir/main.cu.o

CMakeFiles/1dStencil.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/1dStencil.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/1dStencil.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/1dStencil.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/1dStencil.dir/main.cu.o.requires:

.PHONY : CMakeFiles/1dStencil.dir/main.cu.o.requires

CMakeFiles/1dStencil.dir/main.cu.o.provides: CMakeFiles/1dStencil.dir/main.cu.o.requires
	$(MAKE) -f CMakeFiles/1dStencil.dir/build.make CMakeFiles/1dStencil.dir/main.cu.o.provides.build
.PHONY : CMakeFiles/1dStencil.dir/main.cu.o.provides

CMakeFiles/1dStencil.dir/main.cu.o.provides.build: CMakeFiles/1dStencil.dir/main.cu.o


# Object files for target 1dStencil
1dStencil_OBJECTS = \
"CMakeFiles/1dStencil.dir/main.cu.o"

# External object files for target 1dStencil
1dStencil_EXTERNAL_OBJECTS =

CMakeFiles/1dStencil.dir/cmake_device_link.o: CMakeFiles/1dStencil.dir/main.cu.o
CMakeFiles/1dStencil.dir/cmake_device_link.o: CMakeFiles/1dStencil.dir/build.make
CMakeFiles/1dStencil.dir/cmake_device_link.o: CMakeFiles/1dStencil.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.UVC05cgjNz/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/1dStencil.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1dStencil.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1dStencil.dir/build: CMakeFiles/1dStencil.dir/cmake_device_link.o

.PHONY : CMakeFiles/1dStencil.dir/build

# Object files for target 1dStencil
1dStencil_OBJECTS = \
"CMakeFiles/1dStencil.dir/main.cu.o"

# External object files for target 1dStencil
1dStencil_EXTERNAL_OBJECTS =

1dStencil: CMakeFiles/1dStencil.dir/main.cu.o
1dStencil: CMakeFiles/1dStencil.dir/build.make
1dStencil: CMakeFiles/1dStencil.dir/cmake_device_link.o
1dStencil: CMakeFiles/1dStencil.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.UVC05cgjNz/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable 1dStencil"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1dStencil.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1dStencil.dir/build: 1dStencil

.PHONY : CMakeFiles/1dStencil.dir/build

CMakeFiles/1dStencil.dir/requires: CMakeFiles/1dStencil.dir/main.cu.o.requires

.PHONY : CMakeFiles/1dStencil.dir/requires

CMakeFiles/1dStencil.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/1dStencil.dir/cmake_clean.cmake
.PHONY : CMakeFiles/1dStencil.dir/clean

CMakeFiles/1dStencil.dir/depend:
	cd /tmp/tmp.UVC05cgjNz/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.UVC05cgjNz /tmp/tmp.UVC05cgjNz /tmp/tmp.UVC05cgjNz/cmake-build-debug /tmp/tmp.UVC05cgjNz/cmake-build-debug /tmp/tmp.UVC05cgjNz/cmake-build-debug/CMakeFiles/1dStencil.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/1dStencil.dir/depend
