# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/bryan/Documents/dealii-tutorials/indstud

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bryan/Documents/dealii-tutorials/indstud

# Include any dependencies generated for this target.
include CMakeFiles/unit-cell-matrices.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/unit-cell-matrices.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/unit-cell-matrices.dir/flags.make

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o: CMakeFiles/unit-cell-matrices.dir/flags.make
CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o: unit-cell-matrices.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bryan/Documents/dealii-tutorials/indstud/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o -c /home/bryan/Documents/dealii-tutorials/indstud/unit-cell-matrices.cc

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bryan/Documents/dealii-tutorials/indstud/unit-cell-matrices.cc > CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.i

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bryan/Documents/dealii-tutorials/indstud/unit-cell-matrices.cc -o CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.s

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.requires:

.PHONY : CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.requires

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.provides: CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.requires
	$(MAKE) -f CMakeFiles/unit-cell-matrices.dir/build.make CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.provides.build
.PHONY : CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.provides

CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.provides.build: CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o


# Object files for target unit-cell-matrices
unit__cell__matrices_OBJECTS = \
"CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o"

# External object files for target unit-cell-matrices
unit__cell__matrices_EXTERNAL_OBJECTS =

unit-cell-matrices: CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o
unit-cell-matrices: CMakeFiles/unit-cell-matrices.dir/build.make
unit-cell-matrices: /opt/dealii-lib/lib/libdeal_II.g.so.8.5.1
unit-cell-matrices: /usr/lib/liblapack.so
unit-cell-matrices: /usr/lib/libblas.so
unit-cell-matrices: /usr/lib/x86_64-linux-gnu/libz.so
unit-cell-matrices: CMakeFiles/unit-cell-matrices.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bryan/Documents/dealii-tutorials/indstud/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable unit-cell-matrices"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/unit-cell-matrices.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/unit-cell-matrices.dir/build: unit-cell-matrices

.PHONY : CMakeFiles/unit-cell-matrices.dir/build

CMakeFiles/unit-cell-matrices.dir/requires: CMakeFiles/unit-cell-matrices.dir/unit-cell-matrices.cc.o.requires

.PHONY : CMakeFiles/unit-cell-matrices.dir/requires

CMakeFiles/unit-cell-matrices.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/unit-cell-matrices.dir/cmake_clean.cmake
.PHONY : CMakeFiles/unit-cell-matrices.dir/clean

CMakeFiles/unit-cell-matrices.dir/depend:
	cd /home/bryan/Documents/dealii-tutorials/indstud && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bryan/Documents/dealii-tutorials/indstud /home/bryan/Documents/dealii-tutorials/indstud /home/bryan/Documents/dealii-tutorials/indstud /home/bryan/Documents/dealii-tutorials/indstud /home/bryan/Documents/dealii-tutorials/indstud/CMakeFiles/unit-cell-matrices.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/unit-cell-matrices.dir/depend

