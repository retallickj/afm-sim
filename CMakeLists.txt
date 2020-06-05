cmake_minimum_required(VERSION 3.10)

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

if(DEFINED SIQAD_PLUGINS_ROOT)
    set(AFM_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/afmmarcus")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(AFM_INSTALL_DIR "release")
else()
    set(AFM_INSTALL_DIR "debug")
endif()

set(AFM_SYMLINK_DIR ${AFM_INSTALL_DIR})

set(AFM_FILES
    LICENSE
    hopping_animator.physeng
    src/animator.py
    src/channel.py
    src/clocking.py
    src/db-sim-connector.py
    src/hopper.py
    src/lineview.py
    src/model.py
    src/qt_import.py
    src/requirements.txt
)

install(FILES ${AFM_FILES} DESTINATION ${AFM_INSTALL_DIR})
install(FILES src/stylesheets/animator.qss DESTINATION ${AFM_INSTALL_DIR}/stylesheets)

#if(COMMAND symlink_siqadtools AND NOT WIN32)
#    if(DEFINED SYMLINK_INSTALL_PREFIX)
#        set(AFM_SYMLINK_DIR "${SYMLINK_INSTALL_PREFIX}/plugins/afmmarcus")
#    endif()
#    symlink_siqadtools("../.." ${AFM_SYMLINK_DIR})
#endif()
